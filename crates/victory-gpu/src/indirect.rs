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
//! as the title wrote them, is logged with those it was left
//! ([`take_log`]: "ray-tracing indirect dispatch as the title asked" when
//! it ran as asked, else "indirect arguments clamped").
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^16 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more. A world run on an RTX 5070 hung in an indirect
//! dispatch of a loop-free culling program whose arguments another pass
//! writes: 2^20 groups of a heavy program outlast the host's timeout in
//! one dispatch. The budget counts each program's own threads per group,
//! and records the clamp reduces are logged (`take_log`).

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), UAV(u0), UAV(u1), UAV(u2)"
// threads: per group, as the program has them; weight: how many times each
// counts (ray tracing, `ray_weight`); tag: the program's, logged with its
// records.
cbuffer Record : register(b0) {
    uint kind; uint stride; uint records; uint threads; uint tag; uint folding; uint weight; uint unused;
};
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads), at byte 0: past the
// budget, records run empty. At byte 4 the records this list's clamps
// reduced (and ray-tracing ones, reduced or not), the first LOGGED of them
// from byte 16 on (8 words each: tag, kind, the arguments read, those
// left), read back by the host.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 16;
// A one-record dispatch run folded (`folding`): the record count at byte 0
// of `fold`, then up to two records (the fold root constant, then x, y, z)
// from byte 16 on. Up to FOLDED groups of a one-dimensional dispatch run,
// as rows of ROW groups along z and one more dispatch for the rest (the
// translated shader's group is the constant minus one plus x plus z rows).
RWByteAddressBuffer fold : register(u2);
static const uint ROW = 65535;
static const uint FOLDED = 1u << 22;
static const uint LOGGED = 31;
// The cost of `g` groups: their threads, `weight` times over; one group
// count past the whole budget on its own costs past it (no 32-bit overflow).
uint groups_cost(uint3 g) {
    uint groups = g.x * g.y * g.z;
    uint per = max(threads, 1u) * max(weight, 1u);
    return groups > BUDGET / per ? BUDGET + 1 : groups * per;
}
void note(uint3 raw, uint3 left) {
    if (all(raw == left) && weight <= 1) return;
    uint n;
    budget.InterlockedAdd(4, 1, n);
    if (n < LOGGED) {
        uint at = 16 + n * 32;
        budget.Store4(at, uint4(tag, kind, raw.x, raw.y));
        budget.Store4(at + 16, uint4(raw.z, left.x, left.y, left.z));
    }
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint at = id.x * stride;
    uint cost;
    uint3 raw;
    uint3 left;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        raw = uint3(args.Load2(at), 0);
        uint count = min(raw.x, 1u << 24);
        uint instances = min(raw.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        left = uint3(count, instances, 0);
        args.Store2(at, left.xy);
    } else if (kind == 2 && folding != 0) {
        raw = args.Load3(at);
        uint count = 1;
        if (raw.y == 1 && raw.z == 1 && raw.x > ROW && raw.x <= FOLDED) {
            uint rows = raw.x / ROW;
            uint rest = raw.x - rows * ROW;
            fold.Store4(16, uint4(1, ROW, 1, rows));
            fold.Store4(32, uint4(rows * ROW + 1, rest, 1, 1));
            count = rest != 0 ? 2 : 1;
            left = raw;
        } else {
            uint3 g = min(raw, uint3(65535, 65535, 65535));
            if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
                g = uint3(0, 0, 0);
            }
            fold.Store4(16, uint4(0, g));
            left = g;
        }
        cost = groups_cost(left);
        uint before;
        budget.InterlockedAdd(0, cost, before);
        if (before >= BUDGET || before + cost > BUDGET) {
            count = 0;
            left.x = 0;
        }
        fold.Store(0, count);
        note(raw, left);
        return;
    } else if (kind == 2) {
        raw = args.Load3(at);
        uint3 g = min(raw, uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        // Threads per group as the program has them.
        cost = groups_cost(g);
        left = g;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        raw = args.Load3(at);
        uint3 g = uint3(min(raw.x, GROUPS), min(raw.y, 1u), min(raw.z, 1u));
        cost = groups_cost(g);
        left = g;
        args.Store3(at, g);
    }
    uint before;
    budget.InterlockedAdd(0, cost, before);
    bool empty = before >= BUDGET || before + cost > BUDGET;
    if (empty) {
        // Draws keep their layout with no vertices; groups become zero.
        args.Store(at, 0);
        left.x = 0;
    }
    note(raw, left);
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
/// tracing) its records are logged under `tag` ([`take_log`]). Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    (threads, tag, weight): (u32, u32, u32),
    fold: Option<(&ID3D12Resource, u64)>,
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    // Groups of an unknown size count as the console's wave (64) or a
    // mesh group (128).
    let threads = match (threads, kind) {
        (0, 2) => 64,
        (0, _) => 128,
        (n, _) => n,
    };
    let constants = [kind, stride, records, threads, tag, u32::from(fold.is_some()), weight, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        // Unused unless folding: the budget word stands in.
        l.SetComputeRootUnorderedAccessView(3, fold.map_or(budget, |f| f.1));
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
        if let Some((f, _)) = fold {
            l.ResourceBarrier(&[transition(
                f,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
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

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

/// [`done`] for a fold buffer [`bound`] wrote records to.
pub(crate) fn done_fold(list: &HostList, buffer: &ID3D12Resource) {
    done(list, buffer);
}

/// Program hashes by the tag the clamp logs them under.
static TAGS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The tag of the program `hash` (its index among those seen).
pub(crate) fn tag(hash: &str) -> u32 {
    let mut tags = TAGS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(i) = tags.iter().position(|h| h == hash) {
        return i as u32;
    }
    tags.push(hash.to_owned());
    tags.len() as u32 - 1
}

/// Logs the records a list's clamps reduced in its last run, from its
/// budget buffer's words (`words`, read before they are zeroed): the first
/// three per program and kind, then every hundredth, at most 300 lines.
pub(crate) fn take_log(words: &[u32]) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static LINES: AtomicU32 = AtomicU32::new(0);
    static SEEN: std::sync::Mutex<Vec<(u32, u32, u32)>> = std::sync::Mutex::new(Vec::new());
    let Some(&reduced) = words.get(1) else { return };
    if reduced == 0 {
        return;
    }
    let tags = TAGS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for n in 0..(reduced as usize).min(31) {
        let Some(r) = words.get(4 + n * 8..4 + n * 8 + 8) else {
            break;
        };
        let (tag, kind) = (r[0], r[1]);
        {
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            let i = match seen.iter().position(|e| e.0 == tag && e.1 == kind) {
                Some(i) => i,
                None => {
                    seen.push((tag, kind, 0));
                    seen.len() - 1
                }
            };
            seen[i].2 += 1;
            let count = seen[i].2;
            if count > 3 && !count.is_multiple_of(100) {
                continue;
            }
        }
        if LINES.fetch_add(1, Ordering::Relaxed) >= 300 {
            return;
        }
        let shader = tags.get(tag as usize).map_or("?", String::as_str);
        if r[2..5] == r[5..8] {
            // Logged unreduced: a ray-tracing dispatch, as the title asked.
            tracing::info!(
                shader,
                read = ?[r[2], r[3], r[4]],
                "ray-tracing indirect dispatch as the title asked"
            );
            continue;
        }
        tracing::info!(
            shader,
            kind,
            read = ?[r[2], r[3], r[4]],
            left = ?[r[5], r[6], r[7]],
            reduced_in_list = reduced,
            "indirect arguments clamped"
        );
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
