//! End-to-end tests of gathers at a level given at run time
//! (`image_gather4_l`, `image_gather4_l_o`): hand-assembled GFX10 compute
//! programs lowered to DXIL, validated by `dxil.dll` when present, and run
//! on the host GPU and WARP against the footprint computed here.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Texture size and levels: 8x8, 4x4, 2x2, 1x1.
const W: u32 = 8;
const LEVELS: u32 = 4;
/// Every lane gathers around (0.4, 0.4), at level `lane & 3`.
const UV: f32 = 0.4;
/// The `_o` program's texel offsets.
const OFFSET: (i32, i32) = (1, -1);

/// The program: an image descriptor at view-table entry 0, a sampler at
/// sampler-table entry 0, a raw UAV at view-table entry 2; each of 32 lanes
/// gathers red at (UV, UV) at level `lane & 3` (with `offset`, the `_o`
/// form) and stores the four values at dword `4 * lane`.
fn program(offset: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xF40C_0302,
        0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0x7E04_02FF,
        UV.to_bits(), // v_mov_b32 v2, UV
        0x7E06_02FF,
        UV.to_bits(), // v_mov_b32 v3, UV
        0x360A_0083,  // v_and_b32 v5, 3, v0
        0x7E08_0D05,  // v_cvt_f32_u32 v4, v5 (the level)
        0x3420_0084,  // v_lshlrev_b32 v16, 4, v0
    ];
    if offset {
        // x in bits 0-5, y in bits 8-13, signed.
        let word = (OFFSET.0 as u32 & 0x3F) | (OFFSET.1 as u32 & 0x3F) << 8;
        w.extend([0x7E02_02FF, word]); // v_mov_b32 v1, offsets
    }
    w.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
    if offset {
        // image_gather4_l_o v[12:15], v[1:4], s[12:19], s[8:11] dmask:0x1 dim:2D
        w.extend([0xF150_0108, 0x0043_0C01]);
    } else {
        // image_gather4_l v[12:15], v[2:4], s[12:19], s[8:11] dmask:0x1 dim:2D
        w.extend([0xF110_0108, 0x0043_0C02]);
    }
    w.extend([
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE078_1000,
        0x800A_0C10, // buffer_store_dwordx4 v[12:15], v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

/// Lowers a wave32 compute program of 32 threads with the view table at s0
/// and the sampler table at s1.
fn build(words: &[u32]) -> HostShader {
    let table = |user_data: u8| RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    };
    let range = |kind, table_offset_dwords, dwords, user_data| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![table(0), table(1)],
        used: vec![
            range(ResourceType::Srv, 0, 8, 0),
            range(ResourceType::Uav, 16, 4, 0),
            range(ResourceType::Sampler, 0, 4, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        console_cf: false,
        per_lane_only: false,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        stride_table: lowered.stride_table,
        root_uavs: lowered.root_uavs,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Texels of every level, red = x + 8y + 100 * level.
fn level_texels() -> Vec<[f32; 4]> {
    let mut t = Vec::new();
    for n in 0..LEVELS {
        let s = (W >> n).max(1);
        for y in 0..s {
            for x in 0..s {
                t.push([(x + 8 * y + 100 * n) as f32, 0.0, 0.0, 1.0]);
            }
        }
    }
    t
}

/// What lane `l` gathers: the footprint at level `l & 3` whose lower texel
/// is floor(UV * size - 0.5) moved by `offset`, in gather order (x, y+1),
/// (x+1, y+1), (x+1, y), (x, y), through a clamp sampler.
fn want(l: u32, (dx, dy): (i32, i32)) -> [f32; 4] {
    let n = l & 3;
    let s = (W >> n).max(1) as i32;
    let low = |c: i32| (UV * s as f32 - 0.5).floor() as i32 + c;
    let (x, y) = (low(dx), low(dy));
    let red = |i: i32, j: i32| (i.clamp(0, s - 1) + 8 * j.clamp(0, s - 1) + 100 * n as i32) as f32;
    [red(x, y + 1), red(x + 1, y + 1), red(x + 1, y), red(x, y)]
}

/// Runs `shader` on the hardware GPU and on WARP (each skipped without a
/// device) and returns, per one that ran, the 4 floats of each lane.
fn run(shader: &HostShader) -> Vec<(&'static str, Vec<[f32; 4]>)> {
    let texels = level_texels();
    let views = [
        View::Levels(W, W, LEVELS, &texels),
        View::Levels(W, W, LEVELS, &texels),
        View::RawUav(4 * 32),
    ];
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!(u.root, RootType::Table, "{u:?}");
            ud[u.index as usize] = 0;
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let lanes = words[0]
            .chunks(4)
            .map(|c| [0, 1, 2, 3].map(|k| f32::from_bits(c[k])))
            .collect();
        out.push((label, lanes));
    }
    assert!(
        !out.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    out
}

/// Each lane gathers the footprint of its own level, in gather order.
#[test]
fn gathers_at_a_run_time_level_read_that_levels_footprint() {
    for (offset, o) in [(false, (0, 0)), (true, OFFSET)] {
        let shader = build(&program(offset));
        for (label, lanes) in run(&shader) {
            for (l, got) in lanes.iter().enumerate() {
                let want = want(l as u32, o);
                assert_eq!(
                    *got, want,
                    "{label}: lane {l} (offset {offset}): {got:?}, want {want:?}"
                );
            }
        }
    }
}
