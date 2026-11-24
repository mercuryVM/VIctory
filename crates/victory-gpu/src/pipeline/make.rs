//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! shortcut: the budget (half a second of waiting per title frame) is a
//! guess; a pass the title runs only once and whose pipeline takes longer
//! is lost.
//!
//! Built ahead (`VICTORY_PREBUILD=all`): every pipeline the title creates
//! is also queued at creation, behind all first uses, and its host
//! pipeline is made only to be stored in the pipeline library, then let go
//! (the title creates some 60,000; holding them all would hold their
//! driver memory). A later first use loads it from the library instead of
//! compiling it. One run through loading fills the library for the runs
//! after it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    /// Queued to be built ahead (`VICTORY_PREBUILD=all`).
    ahead: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(
            graphics
                .as_ref()
                .map(|g| g.describe())
                .unwrap_or_default()
                .as_bytes(),
        );
        Pending {
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            ahead: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        self.made
            .get_or_init(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }

    /// Makes the host pipeline for the library only, unless a first use
    /// has made or queued it meanwhile.
    fn build_ahead(&self) {
        if self.made.get().is_none() && !self.queued.load(Ordering::Relaxed) {
            drop(HostPipeline::new(&self.shaders, self.graphics.as_deref()));
            BUILT_AHEAD.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Whether every pipeline the title creates is built ahead
/// (`VICTORY_PREBUILD=all`).
fn prebuild_all() -> bool {
    static ALL: OnceLock<bool> = OnceLock::new();
    *ALL.get_or_init(|| {
        let all = std::env::var("VICTORY_PREBUILD").is_ok_and(|v| v.trim() == "all");
        if all {
            tracing::info!("every pipeline the title creates is built ahead");
        }
        all
    })
}

/// Pipelines built ahead, and those queued to be and not built yet.
pub(crate) static BUILT_AHEAD: AtomicU64 = AtomicU64::new(0);
static AHEAD_WAITING: AtomicU64 = AtomicU64::new(0);

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
const BUDGET: Duration = Duration::from_millis(500);

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        start(p);
        wait(p, crate::frame::current());
    }
    let made = p.made.get()?.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
}

/// Queues `p` for the workers, once.
fn start(p: &Arc<Pending>) {
    if !p.queued.swap(true, Ordering::Relaxed) {
        PEAK.fetch_max(
            WAITING.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Relaxed,
        );
        queue(p.clone());
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it),
/// else queues it to be built ahead if every pipeline is.
pub(crate) fn early(p: &Arc<Pending>) {
    if p.shaders.is_empty() {
        return;
    }
    if super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p);
    } else if prebuild_all() && !p.ahead.swap(true, Ordering::Relaxed) {
        AHEAD_WAITING.fetch_add(1, Ordering::Relaxed);
        work().push(Job::Ahead(p.clone()));
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    static WAITED_US: AtomicU64 = AtomicU64::new(0);
    if FRAME.swap(frame, Ordering::Relaxed) != frame {
        WAITED_US.store(0, Ordering::Relaxed);
    }
    let left = BUDGET.saturating_sub(Duration::from_micros(WAITED_US.load(Ordering::Relaxed)));
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    WAITED_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// What the workers do: make a pipeline for its first use, or build one
/// ahead for the library.
enum Job {
    First(Arc<Pending>),
    Ahead(Arc<Pending>),
}

/// The workers' queues: first uses before pipelines built ahead.
struct Queues<T> {
    first: VecDeque<T>,
    ahead: VecDeque<T>,
}

impl<T> Queues<T> {
    const fn new() -> Self {
        Queues {
            first: VecDeque::new(),
            ahead: VecDeque::new(),
        }
    }

    fn pop(&mut self) -> Option<T> {
        self.first.pop_front().or_else(|| self.ahead.pop_front())
    }
}

/// The workers' queues and the signal that work came in.
struct Work {
    queues: Mutex<Queues<Job>>,
    ready: Condvar,
}

impl Work {
    fn push(&self, job: Job) {
        let mut q = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        match job {
            Job::First(_) => q.first.push_back(job),
            Job::Ahead(_) => q.ahead.push_back(job),
        }
        self.ready.notify_one();
    }
}

/// The work queues, with the workers (half the logical processors, 2 to
/// 8) started on first use.
fn work() -> &'static Work {
    static WORK: Work = Work {
        queues: Mutex::new(Queues::new()),
        ready: Condvar::new(),
    };
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        for i in 0..workers {
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(|| {
                    // Below the title's own threads (loading, recording).
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    loop {
                        let job = {
                            let mut q = WORK.queues.lock().unwrap_or_else(|e| e.into_inner());
                            loop {
                                if let Some(job) = q.pop() {
                                    break job;
                                }
                                q = WORK.ready.wait(q).unwrap_or_else(|e| e.into_inner());
                            }
                        };
                        run(job);
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            }
        }
    });
    &WORK
}

/// Runs one job on a worker.
fn run(job: Job) {
    match job {
        Job::First(p) => {
            p.make();
            if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                drained();
            } else {
                super::library::save(false);
            }
        }
        Job::Ahead(p) => {
            p.build_ahead();
            if AHEAD_WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                tracing::info!(
                    built_ahead = BUILT_AHEAD.load(Ordering::Relaxed),
                    "pipelines built ahead: every queued one is done"
                );
                super::library::save(true);
            } else {
                super::library::save(false);
            }
        }
    }
}

/// Hands `p` to the workers for its first use.
fn queue(p: Arc<Pending>) {
    work().push(Job::First(p));
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Workers take every first use before any pipeline built ahead, each
    /// queue in order.
    #[test]
    fn first_uses_go_before_pipelines_built_ahead() {
        let mut q = Queues::new();
        q.ahead.push_back("ahead 1");
        q.ahead.push_back("ahead 2");
        q.first.push_back("first 1");
        assert_eq!(q.pop(), Some("first 1"));
        q.first.push_back("first 2");
        assert_eq!(q.pop(), Some("first 2"));
        assert_eq!(q.pop(), Some("ahead 1"));
        assert_eq!(q.pop(), Some("ahead 2"));
        assert_eq!(q.pop(), None);
    }

    /// A pipeline built ahead is not kept: its first use still makes (or
    /// loads) it; one already queued for a first use is not built ahead.
    #[test]
    fn building_ahead_keeps_nothing() {
        let p = Pending::new(Vec::new(), None);
        p.build_ahead();
        assert!(p.made.get().is_none());
        let before = BUILT_AHEAD.load(Ordering::Relaxed);
        p.queued.store(true, Ordering::Relaxed);
        p.build_ahead();
        assert_eq!(BUILT_AHEAD.load(Ordering::Relaxed), before);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }
}
