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
//! is also queued at creation, below everything else (after requested and
//! early work), and its host pipeline is made only to be stored in the
//! pipeline library, then let go: the title creates some 60,000, and
//! holding them all would hold their driver memory. Afterwards it is not
//! queued again, so a later first use queues it as usual and loads it from
//! the library instead of compiling it. One run through loading fills the
//! library for the runs after it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
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
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        self.made.get_or_init(|| {
            timed("made", &self.shaders, || {
                HostPipeline::new(&self.shaders, self.graphics.as_deref())
            })
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }

    /// Makes the host pipeline for the library only (see the module notes).
    fn build_ahead(&self) {
        if self.made.get().is_none() {
            drop(timed("built ahead", &self.shaders, || {
                HostPipeline::new(&self.shaders, self.graphics.as_deref())
            }));
            BUILT_AHEAD.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A host pipeline taking this long to make is logged (wall time).
const SLOW: Duration = Duration::from_secs(10);

/// Runs `make` (a host pipeline's creation, which translates its shaders
/// first if they are not yet): one taking [`SLOW`] or more is logged with
/// its shaders and this thread's CPU time, to tell a slow driver compile
/// (CPU time near the wall time) from a worker that waited for the CPU.
fn timed<T>(what: &str, shaders: &[Arc<Shader>], make: impl FnOnce() -> T) -> T {
    let (started, cpu) = (Instant::now(), thread_cpu_ms());
    let made = make();
    let wall = started.elapsed();
    if wall >= SLOW {
        let (user, kernel) = thread_cpu_ms();
        let hashes: Vec<String> = shaders
            .iter()
            .map(|s| format!("{} {}", s.stage, s.hash))
            .collect();
        tracing::info!(
            what,
            ms = wall.as_millis() as u64,
            user_ms = user.saturating_sub(cpu.0),
            kernel_ms = kernel.saturating_sub(cpu.1),
            shaders = ?hashes,
            "a host pipeline was slow to make"
        );
    }
    made
}

/// This thread's user and kernel CPU time so far, in ms (zeros off Windows
/// or if the system does not say).
fn thread_cpu_ms() -> (u64, u64) {
    #[cfg(windows)]
    {
        use windows::Win32::Foundation::FILETIME;
        use windows::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
        let (mut created, mut exited, mut kernel, mut user) = (
            FILETIME::default(),
            FILETIME::default(),
            FILETIME::default(),
            FILETIME::default(),
        );
        // SAFETY: the current thread's pseudo-handle and our own out values.
        if unsafe {
            GetThreadTimes(
                GetCurrentThread(),
                &mut created,
                &mut exited,
                &mut kernel,
                &mut user,
            )
        }
        .is_ok()
        {
            let ms = |t: FILETIME| {
                ((u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime)) / 10_000
            };
            return (ms(user), ms(kernel));
        }
    }
    (0, 0)
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

/// Pipelines built ahead, and those queued to be and not taken yet.
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
        start(p, REQUESTED);
        wait(p, crate::frame::current());
    }
    let made = p.made.get()?.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
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
        start(p, EARLY);
    } else if prebuild_all() && workers() {
        let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        if queue.push(p, AHEAD).is_some() {
            AVAILABLE.notify_one();
        }
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

const NOT_QUEUED: u8 = 0;
/// Queued to be built ahead, and taken by a worker to be: both below every
/// first use, so a draw or an early start still queues (promotes) it.
const AHEAD: u8 = 1;
const MAKING_AHEAD: u8 = 2;
const EARLY: u8 = 3;
const REQUESTED: u8 = 4;
const MAKING: u8 = 5;

struct Queue {
    ahead: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
}

/// What a worker took: a pipeline to make for its use, or to build ahead.
#[derive(Debug, PartialEq, Eq)]
enum Job {
    Make,
    Ahead,
}

impl Queue {
    const fn new() -> Self {
        Self {
            ahead: VecDeque::new(),
            early: VecDeque::new(),
            requested: VecDeque::new(),
        }
    }

    /// Queues a pipeline or promotes one queued at a lower priority.
    /// Returns whether it is new to the first-use work (not counted in
    /// `WAITING` yet), or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        match priority {
            REQUESTED => self.requested.push_back(p.clone()),
            EARLY => self.early.push_back(p.clone()),
            _ => {
                self.ahead.push_back(p.clone());
                AHEAD_WAITING.fetch_add(1, Ordering::Relaxed);
            }
        }
        Some(prev < EARLY)
    }

    fn pop(&mut self) -> Option<(Arc<Pending>, Job)> {
        if let Some(p) = self.requested.pop_front().or_else(|| {
            // An early entry can have been promoted and claimed already.
            std::iter::from_fn(|| self.early.pop_front())
                .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        }) {
            p.queue_state.store(MAKING, Ordering::Relaxed);
            return Some((p, Job::Make));
        }
        // An entry to build ahead can have been promoted since.
        while let Some(p) = self.ahead.pop_front() {
            AHEAD_WAITING.fetch_sub(1, Ordering::Relaxed);
            if p.queue_state.load(Ordering::Relaxed) == AHEAD {
                p.queue_state.store(MAKING_AHEAD, Ordering::Relaxed);
                return Some((p, Job::Ahead));
            }
        }
        None
    }

    /// A worker built `p` ahead: unless a first use promoted it meanwhile,
    /// it is not queued any more, so its first use queues it again.
    fn built_ahead(&mut self, p: &Pending) {
        if p.queue_state.load(Ordering::Relaxed) == MAKING_AHEAD {
            p.queue_state.store(NOT_QUEUED, Ordering::Relaxed);
        }
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(
                WAITING.fetch_add(1, Ordering::Relaxed) + 1,
                Ordering::Relaxed,
            );
        }
        AVAILABLE.notify_one();
    }
}

fn next() -> (Arc<Pending>, Job) {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(job) = queue.pop() {
            return job;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// Runs one job a worker took.
fn run(p: Arc<Pending>, job: Job) {
    match job {
        Job::Make => {
            p.make();
            if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                drained();
            } else {
                super::library::save(false);
            }
        }
        Job::Ahead => {
            p.build_ahead();
            QUEUE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .built_ahead(&p);
            if AHEAD_WAITING.load(Ordering::Relaxed) == 0 {
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

/// Starts the workers (half the logical processors, 2 to 8) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        for i in 0..workers {
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(|| {
                    // At the title's own priority: below it, the workers
                    // starved in the world while the title kept every core
                    // busy, and about a thousand pipelines draws wait for
                    // stayed unmade for minutes (RTX 5070, 16 threads).
                    // `VICTORY_PIPELINE_PRIORITY=below` keeps the old one.
                    if std::env::var("VICTORY_PIPELINE_PRIORITY").is_ok_and(|v| v.trim() == "below")
                    {
                        // SAFETY: a plain call on our own thread.
                        unsafe {
                            use windows::Win32::System::Threading::*;
                            let _ =
                                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                        }
                    }
                    loop {
                        let (p, job) = next();
                        run(p, job);
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
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

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop().unwrap().0, &requested));
        assert!(Arc::ptr_eq(&queue.pop().unwrap().0, &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop().unwrap().0, &earlier));
        assert!(queue.pop().is_none());
    }

    /// Pipelines built ahead come after every early and requested one; one
    /// a draw promoted is made for the draw, not built ahead.
    #[test]
    fn pipelines_built_ahead_come_last_and_yield_to_draws() {
        let ahead = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let early = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&ahead, AHEAD), Some(true));
        assert_eq!(queue.push(&promoted, AHEAD), Some(true));
        assert_eq!(queue.push(&early, EARLY), Some(true));
        // A draw promotes a pipeline queued to be built ahead: new to the
        // first-use work.
        assert_eq!(queue.push(&promoted, REQUESTED), Some(true));

        let (p, job) = queue.pop().unwrap();
        assert!(Arc::ptr_eq(&p, &promoted));
        assert_eq!(job, Job::Make);
        let (p, job) = queue.pop().unwrap();
        assert!(Arc::ptr_eq(&p, &early));
        assert_eq!(job, Job::Make);
        // The promoted entry left in the ahead queue is skipped.
        let (p, job) = queue.pop().unwrap();
        assert!(Arc::ptr_eq(&p, &ahead));
        assert_eq!(job, Job::Ahead);
        assert!(queue.pop().is_none());
    }

    /// After a build ahead the pipeline is not queued (its first use
    /// queues it again), unless a draw promoted it while it was built.
    #[test]
    fn a_build_ahead_leaves_the_first_use_to_queue_it() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();
        queue.push(&p, AHEAD);
        assert_eq!(queue.pop().unwrap().1, Job::Ahead);
        p.build_ahead();
        assert!(p.made.get().is_none(), "nothing is kept");
        queue.built_ahead(&p);
        assert_eq!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        assert_eq!(queue.push(&p, REQUESTED), Some(true));

        let q = Arc::new(Pending::new(Vec::new(), None));
        queue.push(&q, AHEAD);
        let _ = queue.pop(); // the requested `p`
        assert_eq!(queue.pop().unwrap().1, Job::Ahead);
        // A draw arrives while `q` is built ahead.
        assert_eq!(queue.push(&q, REQUESTED), Some(true));
        queue.built_ahead(&q);
        assert_eq!(q.queue_state.load(Ordering::Relaxed), REQUESTED);
        assert_eq!(queue.pop().unwrap().1, Job::Make);
    }
}
