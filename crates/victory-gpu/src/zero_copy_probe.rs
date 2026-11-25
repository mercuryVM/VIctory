//! What the host offers for title memory the GPU uses in place instead of
//! copies (zero-copy): GPU upload heaps (VRAM the CPU writes through the
//! resizable BAR) and heaps opened on existing system memory (the title's
//! own pages). Run with
//! `cargo test --release -p victory-gpu --lib -- --ignored --nocapture zero_copy`.

use std::time::Instant;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RESERVE, MapViewOfFile, PAGE_READWRITE,
    SEC_COMMIT, SEC_RESERVE, VirtualAlloc,
};

const MIB: usize = 1 << 20;
const MEM_WRITE_WATCH: u32 = 0x20_0000;

struct Gpu {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: u64,
}

impl Gpu {
    fn new() -> Gpu {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain device creation on the default adapter.
        unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.expect("device");
        let device = device.expect("device");
        // SAFETY: plain object creation.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .unwrap();
            list.Close().unwrap();
            let fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            Gpu { device, queue, allocator, list, fence, value: 0 }
        }
    }

    /// Records with `f`, runs and waits; the GPU time in ms (wall).
    fn run(&mut self, f: impl FnOnce(&ID3D12GraphicsCommandList)) -> f64 {
        // SAFETY: our own list, allocator, queue and fence.
        unsafe {
            self.allocator.Reset().unwrap();
            self.list.Reset(&self.allocator, None).unwrap();
            f(&self.list);
            self.list.Close().unwrap();
            let started = Instant::now();
            self.queue.ExecuteCommandLists(&[Some(self.list.cast().unwrap())]);
            self.value += 1;
            self.queue.Signal(&self.fence, self.value).unwrap();
            while self.fence.GetCompletedValue() < self.value {
                std::thread::yield_now();
            }
            started.elapsed().as_secs_f64() * 1e3
        }
    }

    fn buffer_desc(len: usize, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: len as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        }
    }

    fn committed(&self, kind: D3D12_HEAP_TYPE, len: usize) -> windows::core::Result<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let state = match kind {
            D3D12_HEAP_TYPE_UPLOAD => D3D12_RESOURCE_STATE_GENERIC_READ,
            D3D12_HEAP_TYPE_READBACK => D3D12_RESOURCE_STATE_COPY_DEST,
            _ => D3D12_RESOURCE_STATE_COMMON,
        };
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &Self::buffer_desc(len, D3D12_RESOURCE_FLAG_NONE),
                state,
                None,
                &mut r,
            )
        }?;
        Ok(r.unwrap())
    }

    /// A buffer over all of an opened heap.
    fn placed(&self, heap: &ID3D12Heap) -> windows::core::Result<ID3D12Resource> {
        // SAFETY: a heap of ours.
        let len = unsafe { heap.GetDesc() }.SizeInBytes as usize;
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: plain placed creation at offset 0.
        unsafe {
            self.device.CreatePlacedResource(
                heap,
                0,
                &Self::buffer_desc(len, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut r,
            )
        }?;
        Ok(r.unwrap())
    }
}

fn copy(list: &ID3D12GraphicsCommandList, dst: &ID3D12Resource, src: &ID3D12Resource, len: usize) {
    // SAFETY: buffers of at least `len` bytes.
    unsafe { list.CopyBufferRegion(dst, 0, src, 0, len as u64) };
}

fn fill(at: *mut u8, len: usize, seed: u8) {
    // SAFETY: `len` bytes of ours.
    let s = unsafe { std::slice::from_raw_parts_mut(at, len) };
    for (i, b) in s.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(seed);
    }
}

fn check(at: *const u8, len: usize, seed: u8) -> bool {
    // SAFETY: `len` readable bytes.
    let s = unsafe { std::slice::from_raw_parts(at, len) };
    s.iter().enumerate().all(|(i, &b)| b == (i as u8).wrapping_mul(31).wrapping_add(seed))
}

/// Opens a heap on `len` bytes of system memory at `at` (both ways the
/// API offers), places a buffer over it and checks the GPU reads what the
/// CPU wrote there and the CPU what the GPU wrote; then times GPU copies
/// out of it.
fn try_memory(gpu: &mut Gpu, what: &str, at: *mut u8, len: usize, mapping: Option<HANDLE>) {
    let device3: ID3D12Device3 = gpu.device.cast().expect("ID3D12Device3");
    let mut heaps: Vec<(&str, windows::core::Result<ID3D12Heap>)> = vec![(
        "FromAddress",
        // SAFETY: memory of ours, alive for the test.
        unsafe { device3.OpenExistingHeapFromAddress(at as *const _) },
    )];
    if let Ok(device13) = gpu.device.cast::<ID3D12Device13>() {
        // SAFETY: as above.
        heaps.push(("FromAddress1", unsafe { device13.OpenExistingHeapFromAddress1(at as *const _, len) }));
    } else {
        println!("  {what}: no ID3D12Device13");
    }
    if let Some(m) = mapping {
        // SAFETY: a mapping of ours.
        heaps.push(("FromFileMapping", unsafe { device3.OpenExistingHeapFromFileMapping(m) }));
    }
    for (how, heap) in heaps {
        let heap = match heap {
            Ok(h) => h,
            Err(e) => {
                println!("  {what} {how}: refused: {e}");
                continue;
            }
        };
        // SAFETY: a heap of ours.
        let desc = unsafe { heap.GetDesc() };
        let buffer = match gpu.placed(&heap) {
            Ok(b) => b,
            Err(e) => {
                println!("  {what} {how}: heap of {} MiB, no buffer: {e}", desc.SizeInBytes as usize / MIB);
                continue;
            }
        };
        let n = (desc.SizeInBytes as usize).min(len);
        let vram = gpu.committed(D3D12_HEAP_TYPE_DEFAULT, n).unwrap();
        let back = gpu.committed(D3D12_HEAP_TYPE_READBACK, n).unwrap();
        // CPU -> GPU.
        fill(at, n, 7);
        let read_ms = gpu.run(|l| copy(l, &vram, &buffer, n));
        let read_again = gpu.run(|l| copy(l, &vram, &buffer, n));
        gpu.run(|l| copy(l, &back, &vram, n));
        let mut p = std::ptr::null_mut();
        // SAFETY: a readback buffer of ours.
        unsafe { back.Map(0, None, Some(&mut p)) }.unwrap();
        let gpu_saw = check(p.cast(), n, 7);
        // GPU -> CPU: write the title memory from VRAM.
        let upload = gpu.committed(D3D12_HEAP_TYPE_UPLOAD, n).unwrap();
        let mut u = std::ptr::null_mut();
        // SAFETY: an upload buffer of ours.
        unsafe { upload.Map(0, None, Some(&mut u)) }.unwrap();
        fill(u.cast(), n, 9);
        gpu.run(|l| copy(l, &vram, &upload, n));
        let write_ms = gpu.run(|l| copy(l, &buffer, &vram, n));
        let cpu_saw = check(at, n, 9);
        let mbs = |ms: f64| n as f64 / MIB as f64 / (ms / 1e3);
        println!(
            "  {what} {how}: heap {} MiB (flags {:#x}, pool {:?}, cpu {:?}); GPU saw CPU writes: {gpu_saw}, CPU saw GPU writes: {cpu_saw}; GPU read {:.0} MiB/s ({read_ms:.1}, again {read_again:.1} ms), GPU write {:.0} MiB/s",
            desc.SizeInBytes as usize / MIB,
            desc.Flags.0,
            desc.Properties.MemoryPoolPreference,
            desc.Properties.CPUPageProperty,
            mbs(read_ms.min(read_again)),
            mbs(write_ms),
        );
    }
}

/// What the title's layout needs of heaps on its memory: a big reserved
/// section with a part committed, unmapping a view under an open heap, and
/// many small heaps.
#[test]
#[ignore = "probe"]
fn zero_copy_layout_probe() {
    use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile, VirtualQuery, MEMORY_BASIC_INFORMATION};
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let gpu = Gpu::new();
    let device13: ID3D12Device13 = gpu.device.cast().expect("ID3D12Device13");
    let memory = || {
        let mut c = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
        // SAFETY: our own process's counters.
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        (c.WorkingSetSize / MIB, c.PagefileUsage / MIB)
    };
    const BIG: usize = 4096 * MIB;
    // SAFETY: sections and views of ours.
    unsafe {
        let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE | SEC_RESERVE, (BIG >> 32) as u32, BIG as u32, std::ptr::null());
        assert!(!section.is_null());
        let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BIG).Value.cast::<u8>();
        assert!(!view.is_null());
        // A quarter committed, in the middle.
        VirtualAlloc(view.add(BIG / 2) as *const _, BIG / 4, MEM_COMMIT, PAGE_READWRITE);
        println!("4 GiB reserved view, 1 GiB committed: memory (working set, commit MiB) {:?}", memory());
        let t = Instant::now();
        let whole: windows::core::Result<ID3D12Heap> = device13.OpenExistingHeapFromAddress1(view as *const _, BIG);
        println!("  heap over all 4 GiB: {:?} in {:.1} ms; memory {:?}", whole.as_ref().map(|_| "ok"), t.elapsed().as_secs_f64() * 1e3, memory());
        drop(whole);
        let t = Instant::now();
        let part: windows::core::Result<ID3D12Heap> = device13.OpenExistingHeapFromAddress1(view.add(BIG / 2) as *const _, BIG / 4);
        println!("  heap over the committed 1 GiB: {:?} in {:.1} ms; memory {:?}", part.as_ref().map(|_| "ok"), t.elapsed().as_secs_f64() * 1e3, memory());
        if let Ok(h) = &part {
            let b = gpu.placed(h);
            println!("  buffer over it: {:?}; memory {:?}", b.as_ref().map(|_| "ok"), memory());
        }
        // Unmapping a view a heap is open on.
        let small = 64 * MIB;
        let v2 = MapViewOfFile(section, FILE_MAP_WRITE, 0, (BIG / 2) as u32, small).Value.cast::<u8>();
        let h2: windows::core::Result<ID3D12Heap> = device13.OpenExistingHeapFromAddress1(v2 as *const _, small);
        println!("  second view (alias) heap: {:?}", h2.as_ref().map(|_| "ok"));
        let unmapped = UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: v2.cast() });
        let mut info = MEMORY_BASIC_INFORMATION::default();
        VirtualQuery(v2 as *const _, &mut info, size_of_val(&info));
        println!("  unmap under an open heap: returned {unmapped}, address now state {:#x} type {:#x}", info.State, info.Type);
        drop(h2);
        let mut info = MEMORY_BASIC_INFORMATION::default();
        VirtualQuery(v2 as *const _, &mut info, size_of_val(&info));
        println!("  after the heap is released: state {:#x}", info.State);
        // Many 64 KiB heaps over the committed part.
        let t = Instant::now();
        let mut many = Vec::new();
        for i in 0..2048 {
            match device13.OpenExistingHeapFromAddress1::<ID3D12Heap>(view.add(BIG / 2 + i * 65536) as *const _, 65536) {
                Ok(h) => many.push(h),
                Err(e) => {
                    println!("  64 KiB heap {i} refused: {e}");
                    break;
                }
            }
        }
        println!("  {} heaps of 64 KiB in {:.1} ms; memory {:?}", many.len(), t.elapsed().as_secs_f64() * 1e3, memory());
    }
}

#[test]
#[ignore = "probe"]
fn zero_copy_probe() {
    let mut gpu = Gpu::new();
    let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: the right struct for the feature.
    let ok = unsafe {
        gpu.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            (&raw mut o16).cast(),
            size_of_val(&o16) as u32,
        )
    };
    println!("OPTIONS16: {ok:?}, GPUUploadHeapSupported {}", o16.GPUUploadHeapSupported.as_bool());
    const LEN: usize = 256 * MIB;
    // VRAM baseline and GPU upload heap (CPU write and read speed).
    let a = gpu.committed(D3D12_HEAP_TYPE_DEFAULT, LEN).unwrap();
    let b = gpu.committed(D3D12_HEAP_TYPE_DEFAULT, LEN).unwrap();
    let ms = gpu.run(|l| copy(l, &b, &a, LEN)).min(gpu.run(|l| copy(l, &b, &a, LEN)));
    println!("VRAM -> VRAM copy: {:.0} MiB/s", LEN as f64 / MIB as f64 / (ms / 1e3));
    match gpu.committed(D3D12_HEAP_TYPE_GPU_UPLOAD, LEN) {
        Ok(up) => {
            let mut p = std::ptr::null_mut();
            // SAFETY: a mappable buffer of ours.
            unsafe { up.Map(0, None, Some(&mut p)) }.unwrap();
            let t = Instant::now();
            fill(p.cast(), LEN, 3);
            let w = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut sum = 0u64;
            // SAFETY: the mapping.
            let s = unsafe { std::slice::from_raw_parts(p.cast::<u64>(), 16 * MIB / 8) };
            for &x in s {
                sum = sum.wrapping_add(x);
            }
            let r = t.elapsed().as_secs_f64();
            std::hint::black_box(sum);
            let ms = gpu.run(|l| copy(l, &b, &up, LEN)).min(gpu.run(|l| copy(l, &b, &up, LEN)));
            println!(
                "GPU upload heap: CPU write {:.0} MiB/s, CPU read {:.0} MiB/s, GPU read {:.0} MiB/s",
                256.0 / w,
                16.0 / r,
                LEN as f64 / MIB as f64 / (ms / 1e3)
            );
        }
        Err(e) => println!("GPU upload heap: refused: {e}"),
    }
    // Title-like system memory.
    // SAFETY: fresh allocations and sections of ours, kept for the test.
    unsafe {
        let plain = VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        try_memory(&mut gpu, "private", plain.cast(), LEN, None);
        let watched =
            VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
        try_memory(&mut gpu, "write-watched", watched.cast(), LEN, None);
        for (name, flag) in [("SEC_COMMIT section", SEC_COMMIT), ("SEC_RESERVE section", SEC_RESERVE)] {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE | flag, 0, LEN as u32, std::ptr::null());
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value.cast::<u8>();
            if flag == SEC_RESERVE {
                VirtualAlloc(view as *const _, LEN, MEM_COMMIT, PAGE_READWRITE);
            }
            try_memory(&mut gpu, name, view, LEN, Some(HANDLE(section)));
        }
    }
}
