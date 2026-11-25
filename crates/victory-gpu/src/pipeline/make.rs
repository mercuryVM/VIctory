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
//! Pipelines a draw waits for go ahead of those started early (at the
//! title's creation, because an earlier run drew them).
//!
//! shortcut: the budget (half a second of waiting per title frame) is a
//! guess; a pass the title runs only once and whose pipeline takes longer
//! is lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::host::{GraphicsState, HostPipeline};
use super::{Program, Shader};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    /// Whether it went into the queue of pipelines a draw waits for, and
    /// whether a worker took it (it may be in both queues).
    now: AtomicBool,
    taken: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    pub(crate) key: String,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let key = key(&shaders, graphics.as_deref());
        #[cfg(feature = "diagnostics")]
        log("created", &key, || {
            let state = graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
            format!("{shaders:?}\t{state}")
        });
        Pending {
            key,
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            now: AtomicBool::new(false),
            taken: AtomicBool::new(false),
            made: OnceLock::new(),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        self.made.get_or_init(|| {
            let (started, cpu) = (Instant::now(), super::thread_cpu());
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            #[cfg(feature = "diagnostics")]
            log_built(&self.key);
            let (user, kernel) = super::thread_cpu_since(cpu);
            let ms = started.elapsed().as_millis() as u64;
            // A slow one at info: CPU time near the wall time is a slow
            // driver compile, much below it a worker kept off the CPU.
            if ms >= SLOW_MS {
                tracing::info!(
                    shaders = ?self.shaders,
                    ms,
                    user_ms = user,
                    kernel_ms = kernel,
                    "a host pipeline was slow to make"
                );
            } else {
                tracing::debug!(
                    shaders = ?self.shaders,
                    ms,
                    user_ms = user,
                    kernel_ms = kernel,
                    "pipeline made"
                );
            }
            made
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Wall time from which making a host pipeline is logged at info.
const SLOW_MS: u64 = 10_000;

/// What identifies a title pipeline across runs: its shader containers
/// and graphics state.
pub(crate) fn key(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let mut key = blake3::Hasher::new();
    for s in shaders {
        key.update(s.stage.as_bytes());
        key.update(s.hash.as_bytes());
    }
    key.update(graphics.map(|g| g.describe()).unwrap_or_default().as_bytes());
    key.finalize().to_hex()[..32].to_owned()
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let geometry = p.shaders.iter().any(|s| s.stage == "geometry");
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| s.failed(super::Program::of(s.stage, geometry)))
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

/// Logs the library name of the host pipeline this thread just made or
/// loaded for `key` (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log_built(key: &str) {
    if let Some((name, loaded)) = super::library::LAST.take() {
        log("built", key, || format!("{name}\t{}", if loaded { "loaded" } else { "made" }));
    }
}

/// `VICTORY_PIPELINE_LOG=<file>` (diagnostics): a line per title pipeline
/// key created (`created`, its shaders, its state as described) and first
/// drawn or dispatched with (`used`), and its host pipeline's library name
/// once made or loaded (`built`, also from the pre-build), to compare what
/// a run creates with what the pre-build makes.
#[cfg(feature = "diagnostics")]
fn log(what: &str, key: &str, rest: impl FnOnce() -> String) {
    use std::collections::HashSet;
    use std::io::Write;
    static LOG: Mutex<Option<(std::fs::File, HashSet<String>)>> = Mutex::new(None);
    static PATH: OnceLock<Option<std::ffi::OsString>> = OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_LOG")) else {
        return;
    };
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.is_none() {
        *log = std::fs::File::create(path).ok().map(|f| (f, HashSet::new()));
    }
    let Some((file, seen)) = log.as_mut() else { return };
    if seen.insert(format!("{what} {key}")) {
        let _ = writeln!(file, "{what}\t{key}\t{}", rest());
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
    // Listed for an early start in later runs from its first use, made or
    // not: one a run never got made is the one it should start early.
    if !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
        #[cfg(feature = "diagnostics")]
        log("used", &p.key, String::new);
    }
    if p.made.get().is_none() {
        start(p, true);
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Queues `p` for the workers: once, and once more ahead of the early
/// starts if a draw waits for it (`now`) while it is still queued.
fn start(p: &Arc<Pending>, now: bool) {
    if let Some(now) = enqueue(p, now) {
        queue(p.clone(), now);
    }
}

/// Which queue `p` goes into now (true: the one draws wait for), if any.
fn enqueue(p: &Pending, now: bool) -> Option<bool> {
    let first = !p.queued.swap(true, Ordering::Relaxed);
    if first {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
    }
    let ahead = now && !p.now.swap(true, Ordering::Relaxed);
    (first || ahead && !p.taken.load(Ordering::Relaxed)).then_some(ahead)
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    super::library::note_created(&p.key);
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
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
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    WAITED_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Pipelines for the workers: those draws wait for, then those started
/// early.
static QUEUES: (Mutex<[VecDeque<Arc<Pending>>; 2]>, Condvar) =
    (Mutex::new([VecDeque::new(), VecDeque::new()]), Condvar::new());

/// Hands `p` to the workers (half the logical processors, 2 to 8), started
/// on first use.
fn queue(p: Arc<Pending>, now: bool) {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    let workers = *WORKERS.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        (0..n)
            .filter(|i| {
                let spawned = std::thread::Builder::new()
                    .name(format!("victory-pipelines-{i}"))
                    .spawn({
                        let i = *i;
                        move || work(i)
                    });
                if let Err(e) = &spawned {
                    tracing::warn!(error = %e, "no pipeline worker");
                }
                spawned.is_ok()
            })
            .count()
    });
    if workers > 0 {
        static WATCH: std::sync::Once = std::sync::Once::new();
        WATCH.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-pipelines-watch".into())
                .spawn(|| loop {
                    std::thread::sleep(WATCH_EVERY);
                    report_slow_workers();
                });
        });
    }
    if workers == 0 {
        // No worker: made here instead.
        if !p.taken.swap(true, Ordering::Relaxed) {
            p.make();
            WAITING.fetch_sub(1, Ordering::Relaxed);
        }
        return;
    }
    let (lock, work) = &QUEUES;
    lock.lock().unwrap_or_else(|e| e.into_inner())[usize::from(!now)].push_back(p);
    work.notify_one();
}

/// A worker: makes queued pipelines, those draws wait for first, and saves
/// the cache when idle (first uses of pipelines made before).
fn work(worker: usize) {
    // At the title's own priority: below it, the workers starved in the
    // world while the title kept every core busy, and about a thousand
    // pipelines draws wait for stayed unmade for minutes (RTX 5070, 16
    // threads). `VICTORY_PIPELINE_PRIORITY=below` keeps the old one.
    if std::env::var("VICTORY_PIPELINE_PRIORITY").is_ok_and(|v| v.trim() == "below") {
        // SAFETY: a plain call on our own thread.
        unsafe {
            use windows::Win32::System::Threading::*;
            let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        }
    }
    let (lock, work) = &QUEUES;
    loop {
        let p = {
            let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(p) = queues.iter_mut().find_map(VecDeque::pop_front) {
                    break p;
                }
                let (held, waited) = work
                    .wait_timeout(queues, Duration::from_secs(20))
                    .unwrap_or_else(|e| e.into_inner());
                queues = held;
                if waited.timed_out() {
                    drop(queues);
                    super::library::save(false);
                    queues = lock.lock().unwrap_or_else(|e| e.into_inner());
                }
            }
        };
        if p.taken.swap(true, Ordering::Relaxed) {
            continue; // its other queue entry
        }
        busy(worker, Some((Instant::now(), p.clone())));
        if !translate_own(&p) {
            park(p);
            busy(worker, None);
            continue;
        }
        p.make();
        busy(worker, None);
        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
            drained();
        } else {
            super::library::save(false);
        }
    }
}

/// The translations `p`'s host pipeline needs: each shader's program in a
/// pipeline with or without a geometry stage.
fn programs(p: &Pending) -> impl Iterator<Item = (&Arc<Shader>, Program)> {
    let geometry = p.shaders.iter().any(|s| s.stage == "geometry");
    p.shaders.iter().map(move |s| (s, Program::of(s.stage, geometry)))
}

/// Translates what `p` needs that no other worker is translating; true if
/// all of it is translated now. A worker that waited for another's
/// translation of a shared shader held up the queue: in the world (run15,
/// RTX 5070) ~1,000 pipelines stayed "being made" for 14 minutes while the
/// 8 workers sat behind a few translations of minutes each, and the game
/// used ~4 of 16 cores.
fn translate_own(p: &Pending) -> bool {
    let (mut others, mut translated) = (false, false);
    for (s, program) in programs(p) {
        if s.has_translation(program) {
            continue;
        }
        if s.translating[program as usize].swap(true, Ordering::AcqRel) {
            others = true;
        } else {
            let _ = s.translated_as(program);
            translated = true;
        }
    }
    if translated {
        unpark();
    }
    !others || programs(p).all(|(s, program)| s.has_translation(program))
}

/// Pipelines taken by a worker whose shaders other workers are translating:
/// back at the front of the draws' queue once a translation is done.
/// Locked after `QUEUES`.
static PARKED: Mutex<Vec<Arc<Pending>>> = Mutex::new(Vec::new());

/// Sets `p` aside (still counted in `WAITING`) unless its translations are
/// done by now: asked under the queues' lock, which [`unpark`] takes after a
/// translation, so none is missed.
fn park(p: Arc<Pending>) {
    let (lock, work) = &QUEUES;
    let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
    p.taken.store(false, Ordering::Relaxed);
    if programs(&p).all(|(s, program)| s.has_translation(program)) {
        queues[0].push_front(p);
        work.notify_one();
    } else {
        PARKED.lock().unwrap_or_else(|e| e.into_inner()).push(p);
    }
}

/// A translation is done: parked pipelines go first again.
fn unpark() {
    let (lock, work) = &QUEUES;
    let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
    let mut parked = PARKED.lock().unwrap_or_else(|e| e.into_inner());
    if parked.is_empty() {
        return;
    }
    for p in parked.drain(..).rev() {
        queues[0].push_front(p);
    }
    work.notify_all();
}

/// What each worker is making, and since when.
type Work = (Instant, Arc<Pending>);
static BUSY: Mutex<Vec<Option<Work>>> = Mutex::new(Vec::new());

/// How often the workers are looked at, and how long one may be busy with
/// a pipeline before it is reported (while it is: the line on a slow
/// pipeline comes once it is made).
const WATCH_EVERY: Duration = Duration::from_secs(10);
const BUSY_SLOW: Duration = Duration::from_secs(10);

fn busy(worker: usize, work: Option<Work>) {
    let mut busy = BUSY.lock().unwrap_or_else(|e| e.into_inner());
    if busy.len() <= worker {
        busy.resize_with(worker + 1, || None);
    }
    busy[worker] = work;
}

/// Logs the workers busy with one pipeline for [`BUSY_SLOW`] or longer:
/// its shaders, which translations it lacks yet, and the queues' lengths.
fn report_slow_workers() {
    let slow: Vec<(usize, Work)> = BUSY
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .enumerate()
        .filter_map(|(i, w)| w.clone().map(|w| (i, w)))
        .filter(|(_, (since, _))| since.elapsed() >= BUSY_SLOW)
        .collect();
    if slow.is_empty() {
        return;
    }
    let (now, early) = {
        let q = QUEUES.0.lock().unwrap_or_else(|e| e.into_inner());
        (q[0].len(), q[1].len())
    };
    let parked = PARKED.lock().unwrap_or_else(|e| e.into_inner()).len();
    for (worker, (since, p)) in slow {
        let untranslated: Vec<String> = programs(&p)
            .filter(|(s, program)| !s.has_translation(*program))
            .map(|(s, program)| format!("{} {} {program:?}", s.stage, s.hash))
            .collect();
        tracing::warn!(
            worker,
            seconds = since.elapsed().as_secs(),
            shaders = ?p.shaders,
            ?untranslated,
            waiting = WAITING.load(Ordering::Relaxed),
            queued_for_draws = now,
            queued_early = early,
            parked,
            "pipeline worker slow"
        );
    }
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

    /// An early start is queued once; a draw waiting for it puts it ahead,
    /// once, unless a worker has taken it.
    #[test]
    fn draws_go_ahead_of_early_starts() {
        let p = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&p, false), Some(false));
        assert_eq!(enqueue(&p, false), None);
        assert_eq!(enqueue(&p, true), Some(true));
        assert_eq!(enqueue(&p, true), None);
        let q = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&q, false), Some(false));
        q.taken.store(true, Ordering::Relaxed);
        assert_eq!(enqueue(&q, true), None);
        let r = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&r, true), Some(true));
        assert_eq!(enqueue(&r, false), None);
        // Counted as waiting by `enqueue`, never handed to a worker.
        WAITING.fetch_sub(3, Ordering::Relaxed);
    }

    /// The workers make other pipelines while one's shader is being
    /// translated elsewhere (here: claimed and never finished), and make it
    /// once that translation is done.
    #[test]
    fn workers_do_not_wait_for_another_workers_translation() {
        let stuck = super::super::shader("compute", b"workers: a translation that never ends");
        stuck.translating[Program::Main as usize].store(true, Ordering::Release);
        let blocked = Arc::new(Pending::new(vec![stuck.clone()], None));
        let free: Vec<Arc<Pending>> = (0..32u8)
            .map(|i| {
                let s = super::super::shader("compute", &[b'w', b'f', i]);
                Arc::new(Pending::new(vec![s], None))
            })
            .collect();
        start(&blocked, true);
        for p in &free {
            start(p, true);
        }
        let started = Instant::now();
        while free.iter().any(|p| p.made.get().is_none())
            && started.elapsed() < Duration::from_secs(30)
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(free.iter().all(|p| p.made.get().is_some()), "workers waited");
        assert!(blocked.made.get().is_none(), "made without its translation");
        assert!(
            !stuck.has_translation(Program::Main),
            "translated by a worker that did not claim it"
        );
        // The other worker's translation ends.
        let _ = stuck.translated();
        unpark();
        let started = Instant::now();
        while blocked.made.get().is_none() && started.elapsed() < Duration::from_secs(30) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(blocked.made.get().is_some(), "made once its shader was");
    }

    /// A pipeline parked after its translations were done goes straight
    /// back; one still missing one waits for `unpark`.
    #[test]
    fn parking_checks_the_translations_under_the_lock() {
        // Both are queued as a first use would queue them: workers of other
        // tests may take and make them, and count them off `WAITING`.
        WAITING.fetch_add(2, Ordering::Relaxed);
        let done = super::super::shader("compute", b"parking: done");
        let _ = done.translated();
        let p = Arc::new(Pending::new(vec![done], None));
        p.taken.store(true, Ordering::Relaxed);
        park(p.clone());
        assert!(!p.taken.load(Ordering::Relaxed), "can be taken again");
        assert!(!PARKED.lock().unwrap().iter().any(|q| Arc::ptr_eq(q, &p)));
        let open = super::super::shader("compute", b"parking: open");
        open.translating[Program::Main as usize].store(true, Ordering::Release);
        let q = Arc::new(Pending::new(vec![open.clone()], None));
        park(q.clone());
        assert!(PARKED.lock().unwrap().iter().any(|r| Arc::ptr_eq(r, &q)));
        let _ = open.translated();
        unpark();
        assert!(!PARKED.lock().unwrap().iter().any(|r| Arc::ptr_eq(r, &q)));
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
