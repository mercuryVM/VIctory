//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with (made in time or
//!    not) are listed by a key known when the title creates them (their
//!    shader containers and graphics state); the next run hands those to
//!    the workers at creation, during loading. A full list keeps the
//!    latest run's first.
//!
//! 3. The pre-build (`victory translate --cache --pipelines`) lists the
//!    title pipelines it has put in the library, by the same key, so a
//!    second pre-build skips them without translating or loading anything.
//!    The list holds only for the library bytes it was saved with and for
//!    what names the pipelines (the translator, this module and `host`).
//!
//! The library is kept in segments (`host-pipelines-NNNNN.bin`, and the
//! older single `host-pipelines.bin`): those on disk are loaded as they are,
//! and what a run stores goes into a library of its own, saved as the next
//! segment, after which the run stores into a fresh one. A save writes only
//! what is new: serialising the one library each time rewrote all of it
//! (the pre-build saved 23 times in 27 minutes, up to 387 MB each, ~7 GB
//! on a USB disk, and the machine froze). Saves wait for 10% new (by bytes)
//! and a minute, or ten minutes, or the end of the run; a segment the host
//! refuses (another driver) is deleted.
//!
//! All library calls are serialised: the interface does not promise that
//! stores, loads and serialising may overlap.

use std::collections::HashSet;
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
    device: ID3D12Device1,
    /// Segments loaded or saved, oldest first: each library with the blob it
    /// was made from (it must outlive it; none for one this run saved), its
    /// size and the hash of its bytes.
    saved: Vec<Segment>,
    /// Where this run stores, saved as the next segment.
    current: ID3D12PipelineLibrary1,
    /// Pipelines stored in `current`.
    added: u64,
    /// The next segment's number.
    next: u32,
    /// Title pipelines the pre-build put in the library or found there, by
    /// key, and whether that list changed since the last save.
    prebuilt: HashSet<String>,
    prebuilt_changed: bool,
}

struct Segment {
    library: ID3D12PipelineLibrary1,
    _blob: Vec<u8>,
    bytes: usize,
    hash: blake3::Hash,
}

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

/// The library of earlier versions, loaded first if there.
const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
const PREBUILT_FILE: &str = "prebuilt-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

/// Segment `n`'s file name.
fn segment_file(n: u32) -> String {
    format!("host-pipelines-{n:05}.bin")
}

/// The number of a segment file's name.
fn segment_number(name: &str) -> Option<u32> {
    name.strip_prefix("host-pipelines-")?.strip_suffix(".bin")?.parse().ok()
}

/// The segment files in `dir`, oldest first, with their numbers (the older
/// single file as 0).
fn segment_files(dir: &std::path::Path) -> Vec<(u32, PathBuf)> {
    let mut files: Vec<(u32, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let n = if name == LIBRARY_FILE { 0 } else { segment_number(&name)? };
            Some((n, e.path()))
        })
        .collect();
    files.sort();
    files
}

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let mut saved = Vec::new();
            let mut next = 1;
            let started = std::time::Instant::now();
            for (n, path) in DIR.get().map(|d| segment_files(d)).unwrap_or_default() {
                next = next.max(n + 1);
                let Ok(blob) = std::fs::read(&path) else { continue };
                match open_segment(&device, &blob) {
                    Ok(library) => saved.push(Segment {
                        library,
                        bytes: blob.len(),
                        hash: blake3::hash(&blob),
                        _blob: blob,
                    }),
                    Err(e) => {
                        tracing::info!(error = %e, file = %path.display(), "host pipeline library segment rejected; deleted");
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
            if saved.is_empty() {
                tracing::info!("no host pipeline library yet; starting an empty one");
            } else {
                tracing::info!(
                    segments = saved.len(),
                    bytes = saved.iter().map(|s| s.bytes).sum::<usize>(),
                    ms = started.elapsed().as_millis() as u64,
                    "host pipeline library loaded"
                );
            }
            let Some(current) = empty(&device) else {
                tracing::warn!("no host pipeline library: pipelines are not kept");
                return None;
            };
            let mut library = Library {
                device,
                saved,
                current,
                added: 0,
                next,
                prebuilt: HashSet::new(),
                prebuilt_changed: false,
            };
            // The pre-build's list, if it was saved with these segments.
            if !library.saved.is_empty()
                && let Some(list) = DIR.get().and_then(|d| std::fs::read(d.join(PREBUILT_FILE)).ok())
            {
                library.prebuilt = listed(&list, &tag(&library.digests()));
            }
            Some(Mutex::new(library))
        })
        .as_ref()
}

impl Library {
    /// The saved segments' hashes, in order: what the pre-build's list
    /// holds for.
    fn digests(&self) -> Vec<u8> {
        self.saved.iter().flat_map(|s| *s.hash.as_bytes()).collect()
    }

    /// Bytes saved in segments so far.
    fn saved_bytes(&self) -> usize {
        self.saved.iter().map(|s| s.bytes).sum()
    }
}

/// The library in `blob`, if the host takes it (not after an adapter or
/// driver change, or damage).
fn open_segment(device: &ID3D12Device1, blob: &[u8]) -> windows::core::Result<ID3D12PipelineLibrary1> {
    // SAFETY: the bytes outlive the library (kept beside it).
    unsafe { device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(blob)?.cast() }
}

/// An empty library.
fn empty(device: &ID3D12Device1) -> Option<ID3D12PipelineLibrary1> {
    // An empty library is asked for with no blob at all, which
    // `open_segment` cannot say: an empty slice still has an address, and
    // vkd3d-proton refuses an address with no bytes behind it. That left it
    // with no library at all.
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
        Ok(library) => Some(library),
        Err(e) => {
            tracing::info!(error = %e, "the host refused an empty pipeline library");
            None
        }
    }
}

/// What the pre-build's list holds for: the library's bytes, and what
/// names its pipelines (the translation, as the shader cache keys it; the
/// host descriptions).
fn tag(blob: &[u8]) -> String {
    let limit = victory_shader::emit::loop_limit().to_le_bytes();
    let translator = victory_shader::SOURCE_HASH.as_bytes();
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        translator,
        include_bytes!("../shader_cache.rs"),
        &limit,
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex().to_string()
}

/// The pre-build's list as saved: the tag it holds for, then its keys.
fn list(tag: &str, keys: &HashSet<String>) -> String {
    let keys = keys.iter().map(String::as_str);
    std::iter::once(tag).chain(keys).collect::<Vec<_>>().join("\n")
}

/// The keys of a saved list, if it holds for `tag` (else none).
fn listed(list: &[u8], tag: &str) -> HashSet<String> {
    let list = String::from_utf8_lossy(list);
    let mut lines = list.lines();
    match lines.next() == Some(tag) {
        true => lines.map(str::to_owned).collect(),
        false => HashSet::new(),
    }
}

/// Whether the pre-build has the title pipeline `key` in the library.
pub(crate) fn prebuilt(key: &str) -> bool {
    library().is_some_and(|l| l.lock().unwrap_or_else(|e| e.into_inner()).prebuilt.contains(key))
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(l) = library() {
        let mut l = l.lock().unwrap_or_else(|e| e.into_inner());
        l.prebuilt_changed |= l.prebuilt.insert(key);
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

/// The pipeline named `name`: loaded from a segment of the library by
/// `load` if it is there with the same description, else made by `create`
/// and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), true)));
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    {
        let l = library.lock().unwrap_or_else(|e| e.into_inner());
        let name = PCWSTR(wide.as_ptr());
        if let Some(pso) = l.saved.iter().map(|s| &s.library).chain([&l.current]).find_map(|lib| load(lib, name).ok()) {
            LOADED.fetch_add(1, Ordering::Relaxed);
            return Ok(pso);
        }
    }
    // Compiled outside the lock: compiles take up to seconds.
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), false)));
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { l.current.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        l.added += 1;
    }
    Ok(pso)
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's last pipeline: its name, and whether it was loaded
    /// (else made), for the pipeline log (diagnostics).
    pub(crate) static LAST: std::cell::Cell<Option<(String, bool)>> =
        const { std::cell::Cell::new(None) };
}

/// Saves what is new: the pipelines stored since the last save as the next
/// segment, when there are 10% more bytes than saved and [`MIN_EVERY`] has
/// passed, or [`MAX_EVERY`] has, or `now` (with the early and pre-built
/// lists, which are small, at any save).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else { return };
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let since = last.map_or(std::time::Duration::MAX, |t| t.elapsed());
    if !now && since < MIN_EVERY {
        return;
    }
    let (segment, prebuilt) = {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        let used = USED_CHANGED.load(Ordering::Relaxed);
        if l.added == 0 && !l.prebuilt_changed && !used {
            return;
        }
        let segment = if l.added == 0 {
            None
        } else {
            // SAFETY: a plain query.
            let size = unsafe { l.current.GetSerializedSize() };
            if !due(now, since, size, l.saved_bytes()) {
                return;
            }
            // SAFETY: a buffer of the size the library asks for.
            let bytes = unsafe {
                let mut bytes = vec![0u8; size];
                l.current.Serialize(&mut bytes).map(|()| bytes)
            };
            match bytes {
                Ok(bytes) => Some(bytes),
                Err(e) => {
                    tracing::warn!(error = %e, "host pipeline library not serialised");
                    return;
                }
            }
        };
        let file = l.next;
        let written = match &segment {
            Some(bytes) => std::fs::create_dir_all(dir).and_then(|()| write_atomically(&dir.join(segment_file(file)), bytes)),
            None => Ok(()),
        };
        if let Err(e) = written {
            tracing::warn!(error = %e, "host pipeline library not saved");
            return;
        }
        if let Some(bytes) = &segment {
            // The saved library stays where it is for lookups; this run
            // stores into a fresh one.
            let Some(fresh) = empty(&l.device) else { return };
            let current = std::mem::replace(&mut l.current, fresh);
            l.saved.push(Segment { library: current, _blob: Vec::new(), bytes: bytes.len(), hash: blake3::hash(bytes) });
            l.next += 1;
            l.added = 0;
        }
        l.prebuilt_changed = false;
        USED_CHANGED.store(false, Ordering::Relaxed);
        // Kept beside the segments it holds for (also when a run adds some).
        let prebuilt = (!l.prebuilt.is_empty()).then(|| list(&tag(&l.digests()), &l.prebuilt));
        (segment.map(|b| (file, b.len(), l.saved.len(), l.saved_bytes())), prebuilt)
    };
    *last = Some(std::time::Instant::now());
    drop(last);
    let used = USED.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(Used::list).unwrap_or_default();
    let written = write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()).and_then(|()| match &prebuilt {
        Some(list) => write_atomically(&dir.join(PREBUILT_FILE), list.as_bytes()),
        None => Ok(()),
    });
    let (created, distinct) = created();
    let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
    match written {
        Ok(()) => tracing::info!(
            segment = ?segment.map(|(n, bytes, ..)| (n, bytes)),
            segments = segment.map(|s| s.2),
            total_bytes = segment.map(|s| s.3),
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            distinct,
            shaders_cached,
            shaders_translated,
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "pipeline lists not saved"),
    }
}

/// Whether `new` bytes of pipelines stored `since` the last save are saved
/// now, beside `saved` bytes already in segments: at the end of the run
/// (`now`), else after [`MIN_EVERY`] once they are 10% of what is saved,
/// and after [`MAX_EVERY`] whatever they are.
fn due(now: bool, since: std::time::Duration, new: usize, saved: usize) -> bool {
    now || since >= MAX_EVERY || since >= MIN_EVERY && new * 10 >= saved
}

/// The least time between saves but the last, and the most a save of new
/// pipelines waits for them to reach 10% of what is saved.
const MIN_EVERY: std::time::Duration = std::time::Duration::from_secs(60);
const MAX_EVERY: std::time::Duration = std::time::Duration::from_secs(600);

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipelines created, and their distinct keys.
static CREATED: Mutex<(u64, Option<HashSet<String>>)> = Mutex::new((0, None));

/// The title created a pipeline with `key`.
pub(crate) fn note_created(key: &str) {
    let mut created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    created.0 += 1;
    created.1.get_or_insert_with(HashSet::new).insert(key.to_owned());
}

/// Title pipelines created so far, and how many distinct ones.
fn created() -> (u64, usize) {
    let created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    (created.0, created.1.as_ref().map_or(0, HashSet::len))
}

/// Title pipeline keys drawn or dispatched with: in earlier runs (the
/// list as loaded), and in this one.
#[derive(Default)]
struct Used {
    before: HashSet<String>,
    now: HashSet<String>,
}

impl Used {
    /// The list to keep: this run's keys, then earlier runs' others, at
    /// most `EARLY_MAX` (the oldest go once it is full).
    fn list(&self) -> Vec<String> {
        let earlier = self.before.iter().filter(|k| !self.now.contains(*k));
        self.now.iter().chain(earlier).take(EARLY_MAX).cloned().collect()
    }
}

static USED: Mutex<Option<Used>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Used>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let before: HashSet<String> = listed.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
        tracing::info!(pipelines = before.len(), "title pipelines to start early");
        *used = Some(Used { before, now: HashSet::new() });
    }
    used
}

/// Reads the early list now if not yet, so a save keeps it (a pre-build
/// asks for no title pipeline).
pub(crate) fn keep_early_list() {
    drop(used());
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|u| u.before.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    if used().get_or_insert_with(Used::default).now.insert(key.to_owned()) {
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

    /// This run's keys stay in the list; earlier runs' fill it up to the cap.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys = |r: std::ops::Range<usize>| r.map(|i| format!("{i:032x}")).collect::<HashSet<_>>();
        let used = Used { before: keys(0..EARLY_MAX), now: keys(EARLY_MAX - 10..EARLY_MAX + 10) };
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
    }

    /// The pre-build's list holds only for the library bytes it was
    /// saved with.
    #[test]
    fn the_prebuilt_list_holds_for_its_library_only() {
        let keys: HashSet<String> = ["a".repeat(32), "b".repeat(32)].into();
        let saved = list(&tag(b"library"), &keys);
        assert_eq!(listed(saved.as_bytes(), &tag(b"library")), keys);
        assert!(listed(saved.as_bytes(), &tag(b"library2")).is_empty());
        assert!(listed(b"", &tag(b"library")).is_empty());
    }

    #[test]
    fn a_rejected_blob_is_refused_and_an_empty_library_made() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        assert!(open_segment(&device, &b"not a pipeline library".repeat(16)).is_err());
        let library = empty(&device).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.GetSerializedSize() } > 0);
    }

    /// Segment files are taken in order, the older single file first, and
    /// other files left alone.
    #[test]
    fn segments_are_found_in_order() {
        let dir = tempfile::tempdir().unwrap();
        for name in [segment_file(12), segment_file(2), LIBRARY_FILE.into(), "host-pipelines.tmp".into(), "host-pipelines-x.bin".into()] {
            std::fs::write(dir.path().join(name), b"").unwrap();
        }
        let found: Vec<u32> = segment_files(dir.path()).into_iter().map(|(n, _)| n).collect();
        assert_eq!(found, [0, 2, 12]);
        assert_eq!(segment_number(&segment_file(7)), Some(7));
    }

    /// A pipeline stored in the current library is in the segment saved
    /// from it, found there once loaded; the fresh library after the save
    /// holds nothing (the next save writes only what comes after).
    #[test]
    fn a_saved_segment_holds_its_pipelines_and_the_next_starts_empty() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        let Some(host) = crate::host::host() else { return };
        let device: ID3D12Device1 = host.device.cast().unwrap();
        const SOURCE: &str = "#define RS \"UAV(u0)\"\nRWByteAddressBuffer o : register(u0);\n[RootSignature(RS)][numthreads(1,1,1)] void main() { o.Store(0, 7); }";
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names outlive the call.
        unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("t"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, None)
                .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes.
        let bytes = unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) };
        // SAFETY: a compiled shader with its root signature inside.
        let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, bytes) }.unwrap();
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed interface pointer for the call's duration.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
            ..Default::default()
        };
        let name = HSTRING::from("segment test pipeline");
        let current = empty(&device).unwrap();
        // SAFETY: live descriptions and objects.
        unsafe {
            let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&desc).unwrap();
            current.StorePipeline(PCWSTR(name.as_ptr()), &pso).unwrap();
            let mut blob = vec![0u8; current.GetSerializedSize()];
            current.Serialize(&mut blob).unwrap();
            let segment = open_segment(&device, &blob).unwrap();
            assert!(segment.LoadComputePipeline::<_, ID3D12PipelineState>(PCWSTR(name.as_ptr()), &desc).is_ok());
            let fresh = empty(&device).unwrap();
            assert!(fresh.LoadComputePipeline::<_, ID3D12PipelineState>(PCWSTR(name.as_ptr()), &desc).is_err());
            assert!(fresh.GetSerializedSize() < blob.len(), "the next segment starts empty");
        }
    }

    /// Saves of new pipelines wait for 10% more than is saved and a minute,
    /// or ten minutes; the last save of a run always writes.
    #[test]
    fn saves_wait_for_enough_that_is_new() {
        let s = std::time::Duration::from_secs;
        let mib = 1 << 20;
        // The first segment, after a minute.
        assert!(!due(false, s(30), mib, 0));
        assert!(due(false, s(60), mib, 0));
        // 5% new: not yet, unless ten minutes passed or the run ends.
        assert!(!due(false, s(120), 5 * mib, 100 * mib));
        assert!(due(false, s(600), 5 * mib, 100 * mib));
        assert!(due(true, s(1), 5 * mib, 100 * mib));
        // 10% new after a minute.
        assert!(due(false, s(61), 10 * mib, 100 * mib));
        assert!(!due(false, s(59), 10 * mib, 100 * mib));
    }
}
