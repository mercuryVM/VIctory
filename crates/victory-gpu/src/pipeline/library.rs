//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! All library calls are serialised: the interface does not promise that
//! stores, loads and serialising may overlap.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
}

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most: the most recently
/// drawn with are kept.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR
                .get()
                .and_then(|d| std::fs::read(d.join(LIBRARY_FILE)).ok())
                .unwrap_or_default();
            let library = open(&device, blob);
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                tracing::info!(bytes = blob.len(), "host pipeline library loaded");
                return Some(Library { library, _blob: blob, added: 0 });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // An empty library is asked for with no blob at all, which `make` cannot
    // say: an empty slice still has an address, and vkd3d-proton refuses an
    // address with no bytes behind it. That left it with no library at all.
    let mut raw = std::ptr::null_mut();
    // SAFETY: no blob is read; a success that wrote a pointer wrote an owned
    // ID3D12PipelineLibrary.
    let made = unsafe {
        (Interface::vtable(device).CreatePipelineLibrary)(
            Interface::as_raw(device),
            std::ptr::null(),
            0,
            &ID3D12PipelineLibrary::IID,
            &mut raw,
        )
        .ok()
        .and_then(|()| {
            if raw.is_null() {
                return Err(windows::Win32::Foundation::E_POINTER.into());
            }
            ID3D12PipelineLibrary::from_raw(raw).cast::<ID3D12PipelineLibrary1>()
        })
    };
    match made {
        Ok(library) => Some(Library { library, _blob: Vec::new(), added: 0 }),
        Err(e) => {
            tracing::info!(error = %e, "the host refused an empty pipeline library");
            None
        }
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    if let Ok(pso) = load(&library.lock().unwrap_or_else(|e| e.into_inner()).library, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
    }
    Ok(pso)
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else { return };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    let blob = {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        if l.added == 0 && !USED_CHANGED.swap(false, Ordering::Relaxed) {
            return;
        }
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe {
            let mut bytes = vec![0u8; l.library.GetSerializedSize()];
            l.library.Serialize(&mut bytes).map(|()| bytes)
        };
        match bytes {
            Ok(bytes) => {
                l.added = 0;
                bytes
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                return;
            }
        }
    };
    // Through `used`: an early list not read yet is read before it is
    // written, not replaced by an empty one.
    let (early, listed) = used().as_ref().map(|e| (e.to_text(), e.listed())).unwrap_or_default();
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| write_atomically(&dir.join(LIBRARY_FILE), &blob))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), early.as_bytes()));
    match written {
        Ok(()) => tracing::info!(
            bytes = blob.len(),
            early = listed,
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            built_ahead = super::make::BUILT_AHEAD.load(Ordering::Relaxed),
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipeline keys drawn or dispatched with, each with the last run
/// that did. The list keeps the `EARLY_MAX` most recently used, so the
/// pipelines a run draws with for the first time replace those no run has
/// used for longest instead of being left out once the list is full.
#[derive(Debug)]
struct Early {
    /// This run's number: one past the newest listed.
    run: u64,
    keys: HashMap<String, u64>,
}

impl Early {
    /// The list as saved: `<run> <key>` lines, most recent first. A bare
    /// key (as earlier builds wrote) is older than any numbered one.
    fn parse(text: &str) -> Early {
        let mut keys: HashMap<String, u64> = HashMap::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            let (run, key) = line
                .split_once(' ')
                .and_then(|(run, key)| Some((run.parse().ok()?, key)))
                .unwrap_or((0, line));
            let last = keys.entry(key.to_owned()).or_insert(run);
            *last = (*last).max(run);
        }
        let run = keys.values().max().map_or(1, |r| r + 1);
        let mut early = Early { run, keys };
        if early.keys.len() > EARLY_MAX {
            early.keys = early.newest().map(|(k, r)| (k.clone(), r)).collect();
        }
        early
    }

    /// The keys to keep, most recently used first (ties by key, so the list
    /// is the same for the same uses).
    fn newest(&self) -> impl Iterator<Item = (&String, u64)> {
        let mut keys: Vec<(&String, u64)> = self.keys.iter().map(|(k, &r)| (k, r)).collect();
        keys.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        keys.into_iter().take(EARLY_MAX)
    }

    fn to_text(&self) -> String {
        self.newest().map(|(key, run)| format!("{run} {key}\n")).collect()
    }

    /// Keys the saved list holds.
    fn listed(&self) -> usize {
        self.keys.len().min(EARLY_MAX)
    }

    fn contains(&self, key: &str) -> bool {
        self.keys.contains_key(key)
    }

    /// Notes `key` used this run; whether the saved list changes.
    fn note(&mut self, key: &str) -> bool {
        match self.keys.get_mut(key) {
            Some(run) if *run == self.run => false,
            Some(run) => {
                *run = self.run;
                true
            }
            None => {
                self.keys.insert(key.to_owned(), self.run);
                true
            }
        }
    }
}

static USED: Mutex<Option<Early>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Early>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let early = Early::parse(&listed);
        tracing::info!(pipelines = early.keys.len(), "title pipelines to start early");
        *used = Some(early);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|e| e.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    if used.get_or_insert_with(|| Early::parse("")).note(key) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_full_early_list_takes_new_pipelines_in_place_of_the_oldest() {
        // A list filled by earlier runs: run 1's keys, then run 2's.
        let mut text = String::new();
        for i in 0..EARLY_MAX {
            let run = if i < EARLY_MAX / 2 { 1 } else { 2 };
            text.push_str(&format!("{run} old{i:05}\n"));
        }
        let mut early = Early::parse(&text);
        assert_eq!(early.run, 3);
        assert!(early.contains("old00000"));

        assert!(early.note("new"));
        assert!(early.note(&format!("old{:05}", EARLY_MAX - 1)));
        let kept = Early::parse(&early.to_text());
        assert_eq!(kept.listed(), EARLY_MAX);
        assert!(kept.contains("new"), "a new pipeline is listed though the list was full");
        assert!(kept.contains(&format!("old{:05}", EARLY_MAX / 2)), "run 2 is kept");
        // One of run 1's keys made room.
        let run1 = (0..EARLY_MAX / 2).filter(|i| kept.contains(&format!("old{i:05}"))).count();
        assert_eq!(run1, EARLY_MAX / 2 - 1);
        assert_eq!(kept.run, 4);
    }

    #[test]
    fn early_list_reads_bare_keys_as_oldest() {
        let mut early = Early::parse("aaaa\nbbbb\n\n");
        assert_eq!(early.run, 1);
        assert!(early.contains("aaaa") && early.contains("bbbb"));
        assert!(early.note("aaaa"), "a listed key used again moves to this run");
        assert!(!early.note("aaaa"), "and is noted once per run");
        assert_eq!(early.to_text(), "1 aaaa\n0 bbbb\n");
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }
}
