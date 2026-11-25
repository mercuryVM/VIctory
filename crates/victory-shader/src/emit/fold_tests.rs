//! End-to-end test of folded dispatches (`GROUP_FOLD_ROW`): a synthetic
//! GFX10 compute program that stores its group id, lowered to DXIL,
//! validated by `dxil.dll` when present, and run on the host GPU and WARP
//! with the fold root constant the host sets.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// One thread per group, the table of views at s0 (a raw UAV at entry 0),
/// group x in s1 and group z in s2. Each group stores its x and z at
/// entry `(x + (x >> 16) * 2) & 7` (two dwords): distinct for the groups
/// the cases below run.
const STORE_GROUP_ID: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0201, // v_mov_b32 v1, s1
    0x7E04_0202, // v_mov_b32 v2, s2
    0x2C08_0290, // v_lshrrev_b32 v4, 16, v1
    0xD746_0003,
    0x0405_0304, // v_lshl_add_u32 v3, v4, 1, v1
    0x3606_0687, // v_and_b32 v3, 7, v3
    0x3406_0683, // v_lshlrev_b32 v3, 3, v3
    0xE070_1000,
    0x800A_0103, // buffer_store_dword v1, v3, s[40:43], 0 offen
    0xE070_1004,
    0x800A_0203, // buffer_store_dword v2, v3, s[40:43], 0 offen offset:4
    0xBF81_0000, // s_endpgm
];

fn build() -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        // One user SGPR, then group x (bit 7) and group z (bit 9).
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 1 << 7 | 1 << 9,
            threads: [1, 1, 1],
        }),
        code: STORE_GROUP_ID
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect(),
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
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Groups dispatched, the fold constant, and the (x, z) each entry holds
/// after.
type Case = ([u32; 3], u32, [(u32, u32); 8]);

#[test]
fn folded_dispatches_number_groups_as_the_console() {
    let shader = build();
    // (groups, fold) and what each entry holds after: (x, z) per entry.
    let cases: [Case; 3] = [
        // As given: groups 0..3.
        (
            [4, 1, 1],
            0,
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ],
        ),
        // The rest of a folded dispatch from group 100 on: groups 100..103.
        (
            [4, 1, 1],
            101,
            [
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (100, 0),
                (101, 0),
                (102, 0),
                (103, 0),
            ],
        ),
        // Two rows: groups 0, 1, 65535 and 65536, z read as 0.
        (
            [2, 1, 2],
            1,
            [
                (0, 0),
                (1, 0),
                (65536, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (65535, 0),
            ],
        ),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let fold = Host::fold_param(&root).expect("a compute root signature has the fold");
        for (groups, value, want) in cases {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                assert_eq!(u.root, RootType::Table, "{u:?}");
                ud[u.index as usize] = 0;
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (fold, Root::Constants(vec![value])),
                        ],
                        views: &[View::RawUav(16)],
                        samplers: &[],
                        groups,
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let got: Vec<(u32, u32)> = out[0].chunks(2).map(|c| (c[0], c[1])).collect();
            assert_eq!(got, want, "{label}: groups {groups:?}, fold {value}");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
