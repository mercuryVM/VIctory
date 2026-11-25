//! Title memory the host GPU uses where it is (`VICTORY_ZERO_COPY=1`):
//! instead of a host copy brought up to date before each submission and
//! copied back after ([`crate::shadow`]), the title's own pages are opened as
//! a D3D12 heap (`OpenExistingHeapFromAddress1`) with a buffer placed over
//! it. Shaders read and write title memory itself, over PCIe; the title's
//! writes are there for the GPU and the GPU's for the title, as on the
//! console, with nothing to copy, watch or compare.
//!
//! One heap per allocation or view (a run of committed pages in it), made at
//! the first GPU use of memory in it: opening one pins its pages (1 GiB took
//! 0.5 s on an RTX 5070) and costs ~80 ms even when small. A heap needs
//! committed pages; memory where opening fails (not all committed, a range
//! across allocations) keeps the copy path, as do textures (detiled into
//! copies of their own).
//!
//! The heap holds its pages: the system refuses to unmap or free memory a
//! heap is open on. The runtime asks [`release`] first (views it unmaps,
//! memory the title frees), which drops the heap once nothing holds it.
//! shortcut: a heap something still holds (a recorded list, a descriptor
//! slot, the raw-address page table until its next rewrite) stays, and the
//! unmap or free fails (logged).
//!
//! shortcut: the GPU reads this memory at 25-39 GB/s (VRAM: ~260); passes
//! that read big buffers many times are slower than from a copy in VRAM.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use crate::shadow::Shadow;

/// Whether title memory is used in place (`VICTORY_ZERO_COPY=1`).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("VICTORY_ZERO_COPY").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!("title memory used in place by the host GPU (zero-copy)");
        }
        on
    })
}

/// The most one heap covers: a bigger allocation gets a heap per window of
/// this size it is used in.
const MAX_REGION: usize = 512 << 20;

/// Heaps opened, by their first title address: the buffer's copy object
/// (`Shadow::in_place`) and the heap.
struct Region {
    shadow: Arc<Shadow>,
    _heap: ID3D12Heap,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Region {}

struct Regions {
    by_start: BTreeMap<usize, Region>,
    /// The longest region, to bound the search for one holding an address.
    span: usize,
    /// Allocations (by base) where opening failed: the copy path, not asked
    /// again.
    refused: BTreeSet<usize>,
}

static REGIONS: Mutex<Regions> = Mutex::new(Regions {
    by_start: BTreeMap::new(),
    span: 0,
    refused: BTreeSet::new(),
});

fn regions() -> std::sync::MutexGuard<'static, Regions> {
    REGIONS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts (since the start): heaps opened, their bytes, ms opening them,
/// refused; and per frame (reset by [`log`]): ranges asked for in place and
/// their bytes, on the copy path and their bytes, releases and those that
/// could not drop their heap.
static OPENED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static FRAME: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The title memory `[address, end)` as a buffer over its pages, if it is
/// in a region opened or one that opens now; else `None` (the copy path).
pub(crate) fn find(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let (address, end) = (address as usize, end as usize);
    let found = {
        let r = regions();
        holding(&r, address, end).cloned()
    };
    let found = found.or_else(|| open(address, end));
    match &found {
        Some(_) => {
            FRAME[0].fetch_add(1, Relaxed);
            FRAME[1].fetch_add((end - address) as u64, Relaxed);
        }
        None => {
            FRAME[2].fetch_add(1, Relaxed);
            FRAME[3].fetch_add((end - address) as u64, Relaxed);
        }
    }
    found
}

/// The region holding `[address, end)`, the innermost if several do.
fn holding(r: &Regions, address: usize, end: usize) -> Option<&Arc<Shadow>> {
    let lowest = end.saturating_sub(r.span);
    if lowest > address {
        return None;
    }
    r.by_start
        .range(lowest..=address)
        .rev()
        .map(|(_, region)| &region.shadow)
        .find(|s| {
            let (start, len) = s.range();
            start as usize <= address && end <= (start + len) as usize
        })
}

/// The allocation holding `at`: its base and length, by the allocation query
/// (cheap even in big guarded views, where `VirtualQuery` walks pages).
fn allocation(at: usize) -> Option<(usize, usize)> {
    allocation_kind(at).map(|(base, len, _)| (base, len))
}

/// As [`allocation`], and whether it is a view of a section (the runtime's
/// physical pages), not an image.
fn allocation_kind(at: usize) -> Option<(usize, usize, bool)> {
    use windows_sys::Win32::System::Memory::{
        MemoryRegionInfo, QueryVirtualMemoryInformation, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut region = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: a query about an address; the buffer matches the class.
    let ok = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut region).cast(),
            size_of_val(&region),
            std::ptr::null_mut(),
        )
    };
    // The Windows SDK's bitfields, in declaration order.
    const MAPPED_DATA_FILE: u32 = 1 << 1;
    const MAPPED_PAGE_FILE: u32 = 1 << 3;
    const MAPPED_PHYSICAL: u32 = 1 << 4;
    // SAFETY: filled by the query when it succeeded.
    let view = unsafe { region.Anonymous.Flags }
        & (MAPPED_DATA_FILE | MAPPED_PAGE_FILE | MAPPED_PHYSICAL)
        != 0;
    (ok != 0 && !region.AllocationBase.is_null()).then_some((
        region.AllocationBase as usize,
        region.RegionSize,
        view,
    ))
}

/// Title GPU memory the runtime reserved (`XMemVirtualAlloc`), by base:
/// with views of physical pages, the only memory used in place (not the
/// process heap or the runtime's own, whose frees a heap would block).
static TITLE: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

/// The runtime reserved title GPU memory `[base, base + len)`.
pub(crate) fn allocated(base: usize, len: usize) {
    TITLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(base, len);
}

fn is_title(base: usize) -> bool {
    TITLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(&base)
}

/// The committed pages of `[base, base + len)` around `at`, as one run.
fn committed_run(base: usize, len: usize, at: usize) -> Option<(usize, usize)> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let (mut run, mut cursor, end) = (None::<(usize, usize)>, base, base + len);
    while cursor < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query about an address; nothing is dereferenced.
        if unsafe { VirtualQuery(cursor as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let (from, to) = (
            info.BaseAddress as usize,
            (info.BaseAddress as usize + info.RegionSize).min(end),
        );
        if info.State == MEM_COMMIT {
            run = Some(match run {
                Some((a, b)) if b == from => (a, to),
                _ if from > at => break,
                _ => (from, to),
            });
        } else if from > at {
            break;
        } else {
            run = None;
        }
        cursor = to;
    }
    run.filter(|&(a, b)| a <= at && at < b)
}

/// Opens a heap over the committed run of the allocation holding
/// `[address, end)`, and a buffer over it.
fn open(address: usize, end: usize) -> Option<Arc<Shadow>> {
    const ALIGN: usize = 64 << 10;
    let (base, len, view) = allocation_kind(address)?;
    if end > base + len {
        return None; // across allocations
    }
    if !view && !is_title(base) {
        return None; // not title GPU memory: the copy path
    }
    if regions().refused.contains(&base) {
        return None;
    }
    let refuse = |why: &str| {
        let first = regions().refused.insert(base);
        OPENED[3].fetch_add(1, Relaxed);
        if first {
            tracing::info!(
                base = format_args!("{base:#x}"),
                len,
                address = format_args!("{address:#x}"),
                why,
                "title memory not used in place (copy path)"
            );
        }
        None
    };
    // The allocation, or the window of it holding the range if it is big
    // (a heap pins all its pages).
    let (from, to) = if len <= MAX_REGION {
        (base, base + len)
    } else {
        let window = base + (address - base) / MAX_REGION * MAX_REGION;
        (window, (window + MAX_REGION).min(base + len))
    };
    if end > to {
        return refuse("the range crosses a window of a big allocation");
    }
    let host = crate::host::host()?;
    let device13: ID3D12Device13 = match host.device.cast() {
        Ok(d) => d,
        Err(_) => return refuse("no ID3D12Device13"),
    };
    let started = std::time::Instant::now();
    // SAFETY: committed title memory; the heap is released before the title
    // unmaps or frees it (`release`).
    let opened = |from: usize, to: usize| unsafe {
        device13.OpenExistingHeapFromAddress1::<ID3D12Heap>(from as *const _, to - from)
    };
    // All of it first (a view is committed whole when mapped; a refusal
    // costs ~1 ms), else the committed pages around the range (walking the
    // regions costs ~0.4 ms each in a guarded view).
    let (heap, from, to) = match opened(from, to) {
        Ok(h) => (h, from, to),
        Err(_) => {
            let Some((a, b)) = committed_run(from, to - from, address) else {
                return refuse("not committed");
            };
            let (a, b) = (a.next_multiple_of(ALIGN), b / ALIGN * ALIGN);
            if address < a || end > b {
                return refuse("the range reaches pages not committed");
            }
            match opened(a, b) {
                Ok(h) => (h, a, b),
                Err(e) => return refuse(&format!("heap refused: {e}")),
            }
        }
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: (to - from) as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a buffer over all of the heap.
    if let Err(e) = unsafe {
        host.device.CreatePlacedResource(
            &heap,
            0,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    } {
        return refuse(&format!("buffer refused: {e}"));
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || {
        format!("title memory in place {from:#x}+{:#x}", to - from)
    });
    let ms = started.elapsed().as_millis() as u64;
    OPENED[0].fetch_add(1, Relaxed);
    OPENED[1].fetch_add((to - from) as u64, Relaxed);
    OPENED[2].fetch_add(ms, Relaxed);
    tracing::info!(
        start = format_args!("{from:#x}"),
        len = to - from,
        allocation = format_args!("{base:#x}+{len:#x}"),
        ms,
        heaps = OPENED[0].load(Relaxed),
        pinned_mb = OPENED[1].load(Relaxed) >> 20,
        "title memory used in place"
    );
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Arc::new(Shadow::new_in_place(
        buffer,
        gpu,
        from as u64,
        (to - from) as u64,
    ));
    let mut r = regions();
    // Another thread may have opened one for the same memory meanwhile.
    if let Some(s) = holding(&r, address, end) {
        return Some(s.clone());
    }
    r.span = r.span.max(to - from);
    r.by_start.insert(
        from,
        Region {
            shadow: shadow.clone(),
            _heap: heap,
        },
    );
    drop(r);
    crate::shadow::changed();
    Some(shadow)
}

/// Every region's buffer object (for [`crate::shadow::all`]).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    regions()
        .by_start
        .values()
        .map(|r| r.shadow.clone())
        .collect()
}

/// The title is about to unmap or free `[at, at + len)`: the heaps over it
/// go (they would keep the system from doing so), unless something still
/// holds their buffer (counted, logged once).
pub(crate) fn release(at: usize, len: usize) {
    if enabled() {
        release_now(at, len);
    }
}

fn release_now(at: usize, len: usize) {
    let end = at.saturating_add(len);
    let gone: Vec<(usize, Region)> = {
        let mut r = regions();
        let lowest = at.saturating_sub(r.span);
        let keys: Vec<usize> = r
            .by_start
            .range(lowest..end)
            .filter(|(_, region)| {
                let (start, len) = region.shadow.range();
                (start + len) as usize > at
            })
            .map(|(&k, _)| k)
            .collect();
        let gone = keys
            .into_iter()
            .filter_map(|k| r.by_start.remove(&k).map(|v| (k, v)))
            .collect();
        // The allocation may come back as something else: ask again then.
        r.refused.retain(|&base| base < at || base >= end);
        TITLE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|&base, _| base < at || base >= end);
        gone
    };
    if gone.is_empty() {
        return;
    }
    crate::shadow::changed();
    for (start, region) in gone {
        FRAME[4].fetch_add(1, Relaxed);
        let held = Arc::strong_count(&region.shadow) > 1;
        if held {
            FRAME[5].fetch_add(1, Relaxed);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    start = format_args!("{start:#x}"),
                    "title memory used in place is unmapped or freed while lists or views still hold it: its heap stays, and the system keeps the memory (logged once)"
                )
            });
        }
        OPENED[1].fetch_sub(region.shadow.range().1, Relaxed);
        drop(region);
    }
}

/// The title is about to free `[at, at + len)`, or with `release` the
/// whole allocation at `at`.
pub(crate) fn freeing(at: usize, len: usize, release_all: bool) {
    if !enabled() {
        return;
    }
    match release_all.then(|| allocation(at)).flatten() {
        Some((base, len)) => release_now(base, len),
        None => release_now(at, len),
    }
}

/// Logs the frame's ranges in place and on the copy path, the heaps open
/// and their pinned bytes, over `frames` (with the copy statistics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    if !enabled() {
        return;
    }
    let f: Vec<f64> = FRAME
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let open = regions().by_start.len();
    tracing::info!(
        "zero-copy per frame: {:.0} ranges in place {:.1} MB, {:.0} on the copy path {:.1} MB; {:.1} releases ({:.1} still held); \
         {open} heaps open, {} MB pinned, {} opened in all in {} ms, {} allocations refused",
        f[0],
        f[1] / 1e6,
        f[2],
        f[3] / 1e6,
        f[4],
        f[5],
        OPENED[1].load(Relaxed) >> 20,
        OPENED[0].load(Relaxed),
        OPENED[2].load(Relaxed),
        OPENED[3].load(Relaxed),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const MIB: usize = 1 << 20;
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Runs a copy of `len` bytes from `src` to `dst` on the host GPU and
    /// waits for it.
    fn gpu_copy(dst: &ID3D12Resource, src: &ID3D12Resource, len: usize) {
        let device = &crate::host::host().unwrap().device;
        // SAFETY: plain objects of ours, used once.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .unwrap();
            list.CopyBufferRegion(dst, 0, src, 0, len as u64);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Signal(&fence, 1).unwrap();
            while fence.GetCompletedValue() < 1 {
                std::thread::yield_now();
            }
        }
    }

    fn mapped(kind: D3D12_HEAP_TYPE, len: usize) -> (ID3D12Resource, *mut u8) {
        let device = &crate::host::host().unwrap().device;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let state = if kind == D3D12_HEAP_TYPE_UPLOAD {
            D3D12_RESOURCE_STATE_GENERIC_READ
        } else {
            D3D12_RESOURCE_STATE_COPY_DEST
        };
        let mut r: Option<ID3D12Resource> = None;
        let mut p = std::ptr::null_mut();
        // SAFETY: plain creation and mapping.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
                .unwrap();
            r.as_ref().unwrap().Map(0, None, Some(&mut p)).unwrap();
        }
        (r.unwrap(), p.cast())
    }

    /// Title memory found in place is the title's memory itself: the GPU
    /// reads what the title wrote and the title what the GPU wrote, with
    /// nothing brought up to date or copied back; released, it can be freed.
    #[test]
    fn title_memory_in_place_is_shared_and_released() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let len = 4 * MIB;
        // SAFETY: a fresh write-watched allocation, as the runtime makes.
        let title = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(title, 0);
        // Not the runtime's title memory: the copy path.
        assert!(find(title as u64, title as u64 + 64).is_none());
        allocated(title, len);
        let shadow = find(title as u64 + 4096, title as u64 + 8192).expect("in place");
        assert!(shadow.is_in_place());
        assert!(holding(&regions(), title, usize::MAX).is_none());
        assert_eq!(
            shadow.range(),
            (title as u64, len as u64),
            "the whole allocation"
        );
        assert!(
            Arc::ptr_eq(
                &find(title as u64 + MIB as u64, title as u64 + 2 * MIB as u64).unwrap(),
                &shadow
            ),
            "one heap"
        );
        // The title writes; the GPU reads it, with no refresh.
        // SAFETY: committed memory of ours.
        unsafe { std::ptr::write_bytes(title as *mut u8, 0x5a, len) };
        assert_eq!(shadow.update(title, len), 0, "nothing to bring up to date");
        let (back, b) = mapped(D3D12_HEAP_TYPE_READBACK, len);
        gpu_copy(&back, &shadow.buffer, len);
        assert!(
            // SAFETY: the readback mapping, the copy done.
            unsafe { std::slice::from_raw_parts(b, len) }
                .iter()
                .all(|&x| x == 0x5a)
        );
        // The GPU writes; the title reads it, with no copy-back.
        let (up, u) = mapped(D3D12_HEAP_TYPE_UPLOAD, len);
        // SAFETY: the upload mapping.
        unsafe { std::ptr::write_bytes(u, 0xc3, len) };
        shadow.writing(title, len);
        gpu_copy(&shadow.buffer, &up, len);
        assert_eq!(shadow.copy_back(title, len), 0);
        assert_eq!(shadow.copy_back_page(title), 0);
        assert!(
            // SAFETY: the title's memory.
            unsafe { std::slice::from_raw_parts(title as *const u8, len) }
                .iter()
                .all(|&x| x == 0xc3)
        );
        // While the heap is open the system keeps the memory; released, not.
        drop((back, up));
        drop(shadow);
        release_now(title, len);
        assert!(find_open(title).is_none());
        assert_ne!(
            // SAFETY: our allocation, no heap on it any more.
            unsafe { VirtualFree(title as *mut _, 0, MEM_RELEASE) },
            0,
            "freed once released"
        );
    }

    /// Memory not committed where the range is keeps the copy path, and the
    /// allocation is not asked again.
    #[test]
    fn memory_not_committed_keeps_the_copy_path() {
        if crate::host::host().is_none() {
            return;
        }
        // SAFETY: a fresh reservation with its first half committed.
        let p = unsafe {
            let p = VirtualAlloc(std::ptr::null(), 2 * MIB, MEM_RESERVE, PAGE_READWRITE) as usize;
            VirtualAlloc(p as *const _, MIB, MEM_COMMIT, PAGE_READWRITE);
            p
        };
        allocated(p, 2 * MIB);
        let tail = (p + MIB + 64) as u64;
        assert!(find(tail, tail + 64).is_none());
        assert!(regions().refused.contains(&p));
        // The committed half: refused with its allocation too (asked once).
        assert!(find(p as u64, p as u64 + 64).is_none());
        // SAFETY: ours.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
        release_now(p, 2 * MIB);
        assert!(
            !regions().refused.contains(&p),
            "a freed allocation is asked again if it comes back"
        );
    }

    fn find_open(at: usize) -> Option<Arc<Shadow>> {
        let r = regions();
        holding(&r, at, at + 1).cloned()
    }
}
