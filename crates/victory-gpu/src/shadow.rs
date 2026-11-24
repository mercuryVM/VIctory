//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    /// Kept after the submission as a mirror of the copy (see
    /// [`mirror_bytes`]) when `mirrored`.
    before: Mutex<Option<(usize, Vec<u8>)>>,
    /// Whether `before` outlives submissions: every CPU write to the copy
    /// inside it then also goes into it, and copy-back leaves it equal to
    /// the copy, so it always holds the copy's bytes when no submission
    /// runs, and the next submission declaring those ranges needs no
    /// snapshot. Counted in [`MIRRORED`].
    mirrored: AtomicBool,
}

/// Snapshot memory kept as mirrors between submissions, at most: in the
/// world, shaders write the GPU heap's copy (~180 MB) in most submissions,
/// and snapshotting it each time cost ~15 ms per submission, ~300 ms per
/// frame. `VICTORY_MIRROR_MB` sets it (0: a snapshot per submission).
fn mirror_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        std::env::var("VICTORY_MIRROR_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(512 << 20, |mb| mb << 20)
    })
}

/// Snapshot memory kept as mirrors now.
static MIRRORED: AtomicUsize = AtomicUsize::new(0);

impl Drop for Shadow {
    fn drop(&mut self) {
        let before = self.before.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some((_, bytes)) = before.take() {
            if *self.mirrored.get_mut() {
                MIRRORED.fetch_sub(bytes.capacity(), Ordering::Relaxed);
            }
            give_back(bytes);
        }
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Bytes of idle snapshot memory kept besides the largest buffer.
const POOL_BYTES: usize = 256 << 20;

/// Returns a snapshot buffer to the pool: the largest is always kept (it is
/// the one whose reuse saves millions of page faults), the others while
/// they add up to `POOL_BYTES` (and at most 8 in all).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let mut kept = 0usize;
    let mut n = 0usize;
    pool.retain(|v| {
        n += 1;
        if n == 1 {
            return true;
        }
        kept += v.capacity();
        n <= 8 && kept <= POOL_BYTES
    });
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            mirrored: AtomicBool::new(false),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        // The old buffer is grown in place: allocating a second one while it
        // still holds the world's 174 MB heap doubles the peak on every growth.
        #[cfg(feature = "diagnostics")]
        let had = before.as_ref().map_or(0, |(_, b)| b.len());
        let grown = match before.take() {
            Some((at, mut bytes)) => {
                let (old_end, capacity) = (at + bytes.len(), bytes.capacity());
                // Exactly the growth: the amortized doubling of a Vec this
                // big would double its capacity, and a copying realloc its peak.
                bytes.reserve_exact(end - start - bytes.len());
                if start < at {
                    bytes.splice(0..0, host(start, at).iter().copied());
                }
                if end > old_end {
                    bytes.extend_from_slice(host(old_end.max(at), end));
                }
                if self.mirrored.load(Ordering::Relaxed) {
                    MIRRORED.fetch_add(bytes.capacity() - capacity, Ordering::Relaxed);
                }
                bytes
            }
            None => {
                let mut grown = pooled(end - start);
                grown.extend_from_slice(host(start, end));
                grown
            }
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            crate::copystat::C::SnapshotBytes,
            (grown.len() - had) as u64,
        );
        *before = Some((start, grown));
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        let covered = snapshot
            .as_ref()
            .is_some_and(|(taken, before)| *taken <= from && to <= taken + before.len());
        if !covered && from < to {
            // Never so (each declared range grows the snapshot), but then a
            // mirror would no longer match the copy.
            self.forget_mirror(&mut snapshot);
        }
        if let Some((taken, before)) = &mut *snapshot
            && covered
            && from < to
        {
            let before = &mut before[from - *taken..to - *taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &mut before[at - from..at - from + n];
                let title = &mut title[..n];
                if host != old {
                    if crate::host::read_into((start + at) as *const u8, title) {
                        if merge(host, old, title) {
                            crate::watch::writing(start + at, n);
                            crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                            written += n;
                        }
                        synced.push((start + at) / PAGE);
                    } else {
                        // Unmapped title memory: the GPU's bytes stay in the
                        // copy, as they did; the mirror follows the copy.
                        old.copy_from_slice(host);
                    }
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty()
            && let Some((_, bytes)) = &*snapshot
            && !self.mirrored.load(Ordering::Relaxed)
        {
            // Kept as the copy's mirror while they all fit.
            let n = bytes.capacity();
            let kept = MIRRORED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |m| {
                (m + n <= mirror_bytes()).then_some(m + n)
            });
            if kept.is_ok() {
                self.mirrored.store(true, Ordering::Relaxed);
            } else if let Some((_, bytes)) = snapshot.take() {
                give_back(bytes);
            }
        }
        written
    }

    /// Drops the snapshot, mirror or not.
    fn forget_mirror(&self, snapshot: &mut Option<(usize, Vec<u8>)>) {
        if let Some((_, bytes)) = snapshot.take() {
            if self.mirrored.swap(false, Ordering::Relaxed) {
                MIRRORED.fetch_sub(bytes.capacity(), Ordering::Relaxed);
            }
            give_back(bytes);
        }
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        // The snapshot (a mirror, or this submission's) follows the copy
        // outside the ranges shaders write (refreshes leave those alone):
        // a range declared later in the submission inside it is then
        // compared with what the copy holds, not with older bytes.
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((taken, bytes)) = &mut *before {
            let (a, b) = ((from - start).max(*taken), (to - start).min(*taken + bytes.len()));
            if a < b {
                // SAFETY: inside the mapping, just written.
                let host = unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
                bytes[a - *taken..b - *taken].copy_from_slice(host);
            }
        }
        to - from
    }
}

/// One piece of copy-back: `host` is the copy after the submission, `old`
/// its bytes before, `title` title memory now. The bytes the GPU wrote (the
/// copy's that differ from before) go into `title`, unless the title holds
/// them already; then the copy and `old` take title memory's bytes. True if
/// `title` changed (it is to be written back).
fn merge(host: &mut [u8], old: &mut [u8], title: &mut [u8]) -> bool {
    let mut changed = false;
    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old.iter_mut()) {
        if *h != *o && *t != *h {
            *t = *h;
            changed = true;
        }
        *h = *t;
        *o = *t;
    }
    changed
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The longest title buffer registered so far (it stays after the buffer is
/// released): a buffer starting further below an address than this cannot
/// hold it. Changed under `BUFFERS`' lock.
static MAX_SPAN: AtomicU64 = AtomicU64::new(0);

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let mut map = buffers();
    MAX_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert(
        (start, resource),
        Entry {
            end: start.saturating_add(len),
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The innermost entry holding `[address, end)`: the one starting closest
/// below `address` that reaches `end`, among those starting at most `span`
/// (the longest entry's length) below `end`. Big buffers hold many smaller
/// ones (a heap's placed buffers), so the one holding an address past
/// those can be any number of entries back.
fn enclosing(map: &BTreeMap<(u64, usize), Entry>, span: u64, address: u64, end: u64) -> Option<(u64, usize)> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    map.range((lowest, 0)..=(address, usize::MAX))
        .rev()
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k)
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = enclosing(&map, MAX_SPAN.load(Ordering::Relaxed), address, end);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

/// The longest window in `LOOSE`. Changed under its lock.
static LOOSE_SPAN: AtomicU64 = AtomicU64::new(0);

/// Below this nothing is ever mapped (the null page and its guard area).
const LOWEST_MAPPED: u64 = 0x10000;

/// The window copy holding `[address, end)` among `windows` (by start and
/// length, none longer than `span`), the innermost if several do.
fn enclosing_window<V>(windows: &BTreeMap<(u64, u64), V>, span: u64, address: u64, end: u64) -> Option<&V> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    windows
        .range((lowest, 0)..=(address, u64::MAX))
        .rev()
        .find(|&(&(s, l), _)| s.saturating_add(l) >= end)
        .map(|(_, v)| v)
}

/// The window copy for `[address, end)` outside buffers: one already made
/// that holds it (a range spanning two windows has one of its own, which
/// later ranges in it share), else a new one. None for memory that is not
/// committed (a stale or null address): 4 MiB of zeros would not help.
fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if address < LOWEST_MAPPED {
        return None;
    }
    let start = address & !(WINDOW - 1);
    let len = end.checked_next_multiple_of(WINDOW)? - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = enclosing_window(&map, LOOSE_SPAN.load(Ordering::Relaxed), address, end) {
        return Some(shadow.clone());
    }
    if !crate::texture::committed(address as *const u8, 1) {
        return None;
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    LOOSE_SPAN.fetch_max(len, Ordering::Relaxed);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn entry(start: u64, len: u64) -> Entry {
        Entry {
            end: start + len,
            shadow: None,
        }
    }

    /// A 256 MiB parent with `children` 64 KiB buffers placed in its first
    /// part, as a heap's placed buffers.
    fn heap(base: u64, children: u64) -> BTreeMap<(u64, usize), Entry> {
        let mut map = BTreeMap::new();
        map.insert((base, 1), entry(base, 256 * MIB));
        for i in 0..children {
            map.insert((base + i * 0x10000, 100 + i as usize), entry(base + i * 0x10000, 0x10000));
        }
        map
    }

    /// A copy of fresh write-watched title memory (`pages` 64 KiB pages),
    /// or none without a host GPU. Not freed: the journal keeps its pages.
    fn title_copy(pages: usize) -> Option<(usize, Arc<Shadow>)> {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let len = pages * PAGE;
        // SAFETY: a fresh allocation of ours.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(title, 0);
        let (buffer, data) = host()?.shared_buffer(len as u64)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Some((title, Arc::new(Shadow::new(buffer, data, gpu, title as u64, len as u64))))
    }

    fn poke(at: usize, value: u8) {
        // SAFETY: committed title memory of a test.
        unsafe { (at as *mut u8).write_volatile(value) }
    }

    fn peek(at: usize) -> u8 {
        // SAFETY: committed memory of a test.
        unsafe { (at as *const u8).read_volatile() }
    }

    impl Shadow {
        /// What a shader does: writes the copy.
        fn gpu_poke(&self, offset: usize, value: u8) {
            // SAFETY: inside the mapping.
            unsafe { self.data.add(offset).write_volatile(value) }
        }

        fn host_byte(&self, offset: usize) -> u8 {
            peek(self.data as usize + offset)
        }

        /// One submission whose shaders write `ranges` (offsets), refreshed
        /// and declared as the batch does, `gpu` run in between, copied back.
        fn submit(&self, ranges: &[(usize, usize)], gpu: impl FnOnce()) {
            let start = self.start as usize;
            for &(a, b) in ranges {
                self.update(start + a, b - a);
                self.writing(start + a, b - a);
            }
            gpu();
            for &(a, b) in ranges {
                self.copy_back(start + a, b - a);
            }
        }
    }

    /// What shaders wrote reaches title memory, and the title's own writes
    /// while the GPU ran are kept, also next to the GPU's in a piece; the
    /// snapshot stays as the copy's mirror and serves the next submission.
    #[test]
    fn gpu_writes_reach_the_title_and_title_writes_stay() {
        let Some((title, copy)) = title_copy(4) else { return };
        poke(title + 5, 7);
        copy.submit(&[(0, 4 * PAGE)], || {
            copy.gpu_poke(10, 0xaa);
            poke(title + 11, 0x55); // the title, while the GPU runs
            poke(title + 3 * PAGE, 0x66);
        });
        assert_eq!((peek(title + 5), peek(title + 10), peek(title + 11)), (7, 0xaa, 0x55));
        assert_eq!(peek(title + 3 * PAGE), 0x66);
        assert_eq!((copy.host_byte(10), copy.host_byte(11)), (0xaa, 0x55));
        assert!(copy.mirrored.load(Ordering::Relaxed), "kept as the mirror");
        copy.submit(&[(0, 4 * PAGE)], || copy.gpu_poke(2 * PAGE, 0xbb));
        assert_eq!(
            (peek(title + 2 * PAGE), peek(title + 10), peek(title + 11)),
            (0xbb, 0xaa, 0x55)
        );
        assert_eq!(copy.host_byte(3 * PAGE), 0x66, "refreshed");
    }

    /// The title writes a byte, a refresh copies it, the title writes it
    /// again while the GPU runs (which does not write it): its last write
    /// stays. A mirror missing the refresh would take the copy's byte for
    /// the GPU's and put it back over the title's.
    #[test]
    fn a_mirror_follows_refreshes() {
        let Some((title, copy)) = title_copy(2) else { return };
        copy.submit(&[(0, 2 * PAGE)], || {});
        assert!(copy.mirrored.load(Ordering::Relaxed));
        poke(title + 200, 0x11);
        copy.submit(&[(0, 2 * PAGE)], || {
            assert_eq!(copy.host_byte(200), 0x11, "refreshed");
            poke(title + 200, 0x33);
        });
        assert_eq!(peek(title + 200), 0x33, "the title's write was lost");
        copy.update(title, 2 * PAGE);
        assert_eq!(copy.host_byte(200), 0x33, "taken at the next refresh");
    }

    /// In one submission: ranges declared at either end, then the range
    /// between them refreshed (after a title write) and declared. The
    /// snapshot already spans it and must hold what the refresh copied.
    #[test]
    fn a_range_declared_inside_the_snapshot_is_compared_with_its_refresh() {
        let Some((title, copy)) = title_copy(3) else { return };
        copy.update(title, 3 * PAGE);
        copy.writing(title, PAGE);
        copy.writing(title + 2 * PAGE, PAGE);
        poke(title + PAGE + 40, 0x21);
        copy.update(title + PAGE, PAGE);
        assert_eq!(copy.host_byte(PAGE + 40), 0x21);
        copy.writing(title + PAGE, PAGE);
        poke(title + PAGE + 40, 0x42); // the title, while the GPU runs
        for at in [title, title + 2 * PAGE, title + PAGE] {
            copy.copy_back(at, PAGE);
        }
        assert_eq!(peek(title + PAGE + 40), 0x42, "the title's write was lost");
    }

    /// A mirror is counted while its copy lives.
    #[test]
    fn mirrors_are_counted_while_their_copy_lives() {
        let Some((_, copy)) = title_copy(2) else { return };
        copy.submit(&[(0, 2 * PAGE)], || {});
        assert!(copy.mirrored.load(Ordering::Relaxed));
        let n = copy.before.lock().unwrap().as_ref().unwrap().1.capacity();
        assert!(n >= 2 * PAGE);
        assert!(MIRRORED.load(Ordering::Relaxed) >= n);
        drop(copy);
    }

    #[test]
    fn merge_takes_only_what_the_gpu_wrote() {
        // The GPU wrote byte 1, the title byte 2.
        let (mut host, mut old, mut title) = ([1, 9, 3, 4], [1, 2, 3, 4], [1, 2, 8, 4]);
        assert!(merge(&mut host, &mut old, &mut title));
        assert_eq!((host, old, title), ([1, 9, 8, 4], [1, 9, 8, 4], [1, 9, 8, 4]));
        // The title holds what the GPU wrote already: nothing to write.
        let (mut host, mut old, mut title) = ([5, 6], [0, 6], [5, 7]);
        assert!(!merge(&mut host, &mut old, &mut title));
        assert_eq!((host, old), ([5, 7], [5, 7]));
    }

    #[test]
    fn a_parent_is_found_past_many_children() {
        let base = 0x10_0000_0000;
        let map = heap(base, 200);
        let span = 256 * MIB;
        // Past all 200 children (and the 64 entries looked at before).
        let at = base + 200 * 0x10000 + 0x100;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        let first_64 = map.range(..=(at, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= at + 16);
        assert!(first_64.is_none(), "the parent is more than 64 entries back");
        // Near the parent's end.
        let at = base + 256 * MIB - 16;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        // Inside a child: the child, the innermost.
        let at = base + 150 * 0x10000 + 8;
        assert_eq!(enclosing(&map, span, at, at + 8), Some((base + 150 * 0x10000, 250)));
        // Across two children: only the parent holds it.
        let at = base + 151 * 0x10000 - 8;
        assert_eq!(enclosing(&map, span, at, at + 16), Some((base, 1)));
        // Past the parent, and before it.
        let at = base + 256 * MIB;
        assert_eq!(enclosing(&map, span, at, at + 1), None);
        assert_eq!(enclosing(&map, span, base - 16, base + 16), None);
        // Longer than any buffer: none, without looking.
        assert_eq!(enclosing(&map, span, base, base + span + 1), None);
        assert_eq!(enclosing(&map, 0, base, base + 1), None);
    }

    #[test]
    fn registered_buffers_widen_the_search() {
        // High above anything a test maps; resource numbers no live
        // resource has.
        let base = 0x7f00_0000_0000;
        let len = 512 * MIB;
        register(1, base, len);
        for i in 0..100u64 {
            register(2 + i as usize, base + i * 0x1000, 0x1000);
        }
        assert!(MAX_SPAN.load(Ordering::Relaxed) >= len);
        let at = base + len - 64;
        let found = enclosing(&buffers(), MAX_SPAN.load(Ordering::Relaxed), at, at + 64);
        for i in 0..100u64 {
            unregister(2 + i as usize, base + i * 0x1000);
        }
        unregister(1, base);
        assert_eq!(found, Some((base, 1)));
        assert!(buffers().range((base, 0)..=(base + len, usize::MAX)).next().is_none());
    }

    #[test]
    fn loose_windows_are_reused_for_ranges_inside_them() {
        let w = WINDOW;
        let base = 0x20_0000_0000;
        let mut windows = BTreeMap::new();
        windows.insert((base, w), "first");
        // A range across the next two windows got one of its own.
        windows.insert((base + w, 2 * w), "spanning");
        let span = 2 * w;
        let find = |address: u64, len: u64| enclosing_window(&windows, span, address, address + len).copied();
        assert_eq!(find(base, 1), Some("first"));
        assert_eq!(find(base + w - 16, 16), Some("first"));
        // Inside the spanning window, also in its second half.
        assert_eq!(find(base + w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 2 * w + 8, 64), Some("spanning"));
        assert_eq!(find(base + 3 * w - 8, 8), Some("spanning"));
        // Across the first window's end: none holds it whole.
        assert_eq!(find(base + w - 8, 16), None);
        // Past every window, before them, longer than any.
        assert_eq!(find(base + 3 * w, 1), None);
        assert_eq!(find(base - 8, 4), None);
        assert_eq!(find(base, 3 * w), None);
    }

    #[test]
    fn loose_memory_must_be_committed() {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // The null page and its guard area.
        assert!(loose(0, 16).is_none());
        assert!(loose(0x1000, 0x1010).is_none());
        assert!(loose(LOWEST_MAPPED - 1, LOWEST_MAPPED + 1).is_none());
        // Reserved, not committed. Not where an earlier test's window was
        // (windows are never freed; other tests' memory is).
        let mut kept = Vec::new();
        let p = loop {
            // SAFETY: a fresh reservation, released below.
            let p = unsafe { VirtualAlloc(std::ptr::null(), WINDOW as usize, MEM_RESERVE, PAGE_READWRITE) } as u64;
            assert_ne!(p, 0);
            let windows = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
            let span = LOOSE_SPAN.load(Ordering::Relaxed);
            if enclosing_window(&windows, span, p + 64, p + 128).is_none() {
                break p;
            }
            kept.push(p);
        };
        assert!(loose(p + 64, p + 128).is_none());
        // SAFETY: our reservations.
        unsafe {
            for p in kept.into_iter().chain([p]) {
                VirtualFree(p as *mut _, 0, MEM_RELEASE);
            }
        }
        // Too close to the top to round up to a window.
        assert!(loose(u64::MAX - 8, u64::MAX).is_none());
    }
}
