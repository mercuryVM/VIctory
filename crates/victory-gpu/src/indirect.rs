//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! A dispatch of a program that intersects rays with a BVH counts its
//! threads [`ray_weight`] times (`VICTORY_RAY_WEIGHT`, default 64): its
//! traversal loops, emulated node by node and bounded only by the loop
//! limit, cost hundreds of other threads each, and one full-screen
//! dispatch of them whose rays never end (nodes the host has not mirrored)
//! runs until the driver resets the device. Each such record's arguments,
//! as the title wrote them, and whether it ran empty, are logged (the first
//! [`LOGGED`], then one in [`LOG_EVERY`]).
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=5, b0), UAV(u0), UAV(u1)"
// weight: threads count that many times (ray tracing); tag: the program's
// hash, logged with its records.
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint weight; uint tag; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty. Then a count of logged records and, from byte 16,
// up to 15 of them: tag, x, y, z (bit 31: ran empty).
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint at = id.x * stride;
    uint cost;
    uint3 asked = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        asked = args.Load3(at);
        uint3 g = min(asked, uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        uint groups = g.x * g.y * g.z;
        uint w = max(weight, 1u);
        // Over the whole budget on its own: past it (no 32-bit overflow).
        cost = groups > BUDGET / (64 * w) ? BUDGET + 1 : groups * 64 * w;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
    }
    uint before;
    budget.InterlockedAdd(0, cost, before);
    bool empty = before >= BUDGET || before + cost > BUDGET;
    if (empty) {
        // Draws keep their layout with no vertices; groups become zero.
        args.Store(at, 0);
    }
    if (kind == 2 && weight > 1) {
        uint n;
        budget.InterlockedAdd(4, 1, n);
        if (n < 15) {
            uint z = min(asked.z, 0x7fffffffu) | (empty ? 0x80000000u : 0u);
            budget.Store4(16 + 16 * n, uint4(tag, asked.x, asked.y, z));
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. A
/// dispatch's threads count `weight` times; with a weight over 1 (ray
/// tracing) its records are logged under `tag` ([`log_records`]). Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    (weight, tag): (u32, u32),
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, weight, tag];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// After the execution: the buffer back to the common state, which title
/// buffers' host copies are kept in, so the list's later uses of it (vertex
/// data, shader reads and writes) promote it again. Left in unordered
/// access, a later vertex or shader-resource read in the list would find it
/// in a state that does not allow that read.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// How many times a ray-tracing dispatch's threads count against the
/// list's budget: `VICTORY_RAY_WEIGHT` (1 counts them as other threads),
/// else 64.
pub(crate) fn ray_weight() -> u32 {
    static WEIGHT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *WEIGHT.get_or_init(|| {
        std::env::var("VICTORY_RAY_WEIGHT")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map_or(64, |w| w.clamp(1, 1 << 16))
    })
}

/// Ray-tracing records logged one by one; past them, one in [`LOG_EVERY`].
pub(crate) const LOGGED: u64 = 64;
pub(crate) const LOG_EVERY: u64 = 256;

/// The ray-tracing records the clamp logged in a list's budget words
/// (`words`: its 64 mapped dwords) when it last ran, logged here (see
/// [`LOGGED`]); clears the count.
///
/// # Safety
/// `words` is the list's mapped budget buffer, and the list is not running.
pub(crate) unsafe fn log_records(words: *mut u32) {
    static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    // SAFETY: the caller's mapped words, 64 of them.
    let n = unsafe { words.add(1).read_volatile() };
    for k in 0..n.min(15) as usize {
        // SAFETY: as above; record k is words 4 + 4k to 7 + 4k.
        let r: [u32; 4] =
            std::array::from_fn(|j| unsafe { words.add(4 + 4 * k + j).read_volatile() });
        let seen = SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if seen < LOGGED || seen.is_multiple_of(LOG_EVERY) {
            tracing::info!(
                tag = format!("{:08x}", r[0]),
                x = r[1],
                y = r[2],
                z = r[3] & 0x7fff_ffff,
                ran_empty = r[3] >> 31 != 0,
                logged = seen + 1,
                "ray-tracing indirect dispatch as the title asked"
            );
        }
    }
    if n > 15 {
        tracing::debug!(
            unlogged = n - 15,
            "ray-tracing indirect dispatches past the log"
        );
    }
    // SAFETY: as above.
    unsafe { words.add(1).write_volatile(0) };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }
}
