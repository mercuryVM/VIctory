//! Registers the compiler reuses for a lane mask on one path and data on
//! another (read as data, or joined as a mask), and the SCC an EXEC branch
//! decides: hand-assembled GFX10 compute programs (wave32, 32
//! lanes) lowered to DXIL, validated by `dxil.dll` when present, and run on
//! the host GPU against what the console computes.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const VCC_LO: u32 = 106;
const EXEC_LO: u32 = 126;
const LITERAL: u32 = 255;

const fn int(n: u32) -> u32 {
    128 + n
}

const fn vreg(n: u32) -> u32 {
    256 + n
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}

const fn sopc(op: u32, src0: u32, src1: u32) -> u32 {
    0xBF00_0000 | op << 16 | src1 << 8 | src0
}

/// A SOPP branch from dword `at` to dword `to`.
const fn branch(op: u32, at: usize, to: usize) -> u32 {
    let delta = to as i32 - at as i32 - 1;
    0xBF80_0000 | op << 16 | (delta as u32 & 0xFFFF)
}

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}

const S_MOV_B32: u32 = 0x03;
const S_ANDN2_WREXEC_B32: u32 = 0x47;
const S_ADD_U32: u32 = 0x00;
const S_CSELECT_B32: u32 = 0x0A;
const S_AND_B32: u32 = 0x0E;
const S_CMP_EQ_U32: u32 = 0x06;
const S_CMP_LG_U32: u32 = 0x07;
const S_CMP_LT_U32: u32 = 0x0A;
const S_CBRANCH_SCC1: u32 = 0x05;
const S_CBRANCH_EXECNZ: u32 = 0x09;
const V_MOV_B32: u32 = 0x01;
const V_READFIRSTLANE_B32: u32 = 0x02;
const V_AND_B32: u32 = 0x1B;
const V_CMP_GT_U32: u32 = 0xC4;
const V_CMPX_EQ_U32: u32 = 0xD2;

/// The table at s0 into s[40:43] and each lane's byte offset into v16.
const PROLOGUE: [u32; 5] = [
    0xBE84_0300, // s_mov_b32 s4, s0 (the table)
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
];

/// Each lane stores v2 at dword `lane` of the raw UAV, then the end.
const EPILOGUE: [u32; 4] = [
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000,
    0x800A_0210, // buffer_store_dword v2, v16, s[40:43], 0 offen
    0xBF81_0000, // s_endpgm
];

/// `v_mbcnt_lo_u32_b32 v3, -1, 0`: code that reads lanes, so the program
/// runs with the console's lanes.
const MBCNT: [u32; 2] = [0xD765_0003, 0x0001_00C1];

/// Lowers a wave32 compute program of 32 threads with a table holding one
/// raw UAV at s0, validated when `dxil.dll` is found.
fn lower_cs(words: &[u32]) -> Result<HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
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
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
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
    })?;
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
    }
    Ok(shader)
}

fn program(body: &[u32]) -> Vec<u32> {
    let mut w = PROLOGUE.to_vec();
    w.extend(body);
    w.extend(EPILOGUE);
    w
}

/// Runs a wave32 program of 32 threads on the hardware device and returns
/// the 32 dwords its lanes store (`None` with no SM 6.6 hardware device).
fn run(words: &[u32]) -> Option<Vec<u32>> {
    let shader = lower_cs(words).unwrap_or_else(|e| panic!("lowers: {e:?}"));
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // The console's wave size (32), which WARP's pipeline creation refuses.
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return None;
    };
    let (root, pso) = host
        .compute_pipeline(&shader)
        .unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let words = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(32)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(words[0].clone())
}

/// `s6` holds the mask of lanes below 16 (`v_cmp` then `s_mov s6, vcc_lo`)
/// unless a uniform branch (`s5 == k`, s5 being 0) skips nothing and the
/// number 7 replaces it; after the join, `v_mov_b32 v2, s6` reads it as
/// data.
fn mask_or_data(k: u32) -> Vec<u32> {
    program(&[
        vopc(V_CMP_GT_U32, int(16), 0), // 5: vcc = lanes < 16
        sop1(S_MOV_B32, 6, VCC_LO),     // 6: s6 = vcc_lo
        sopc(S_CMP_EQ_U32, 5, int(k)),  // 7
        branch(S_CBRANCH_SCC1, 8, 10),  // 8
        sop1(S_MOV_B32, 6, int(7)),     // 9: s6 = 7
        vop1(V_MOV_B32, 2, 6),          // 10: v2 = s6
    ])
}

/// The join of a mask and data, read as data, is the console's number on
/// whichever path the wave took.
#[test]
fn a_register_joining_a_mask_and_data_reads_as_its_number() {
    for (k, want) in [(0, 0xFFFF), (1, 7)] {
        let Some(got) = run(&mask_or_data(k)) else {
            return;
        };
        assert_eq!(got, vec![want; 32], "s5 == {k}");
    }
}

/// A waterfall loop over v1 = lane & 3 (`v_readfirstlane`, `v_cmpx_eq`,
/// `s_andn2_wrexec_b32 s28, s28`, `s_cbranch_execnz`), then EXEC restored
/// and `s_cselect_b32 s5, exec_lo, 0` on the SCC the last `s_andn2_wrexec`
/// left: zero, for the loop only ends once no lane is left. `v2 = s5`.
fn after_a_waterfall() -> Vec<u32> {
    program(&[
        vop2(V_AND_B32, 1, int(3), 0),           // 5: v1 = lane & 3
        sop1(S_MOV_B32, 3, EXEC_LO),             // 6: s3 = exec
        sop1(S_MOV_B32, 28, EXEC_LO),            // 7: s28 = exec
        vop1(V_READFIRSTLANE_B32, 29, vreg(1)),  // 8: s29 = first v1
        vopc(V_CMPX_EQ_U32, 29, 1),              // 9: exec = v1 == s29
        sop1(S_ANDN2_WREXEC_B32, 28, 28),        // 10
        branch(S_CBRANCH_EXECNZ, 11, 8),         // 11
        sop1(S_MOV_B32, EXEC_LO, 3),             // 12: exec = s3
        sop2(S_CSELECT_B32, 5, EXEC_LO, int(0)), // 13: s5 = scc ? exec : 0
        vop1(V_MOV_B32, 2, 5),                   // 14: v2 = s5
    ])
}

/// The SCC of the EXEC a branch tests is known on each of its edges: zero
/// where a waterfall loop is left.
#[test]
fn the_scc_a_waterfall_leaves_is_zero_after_it() {
    let Some(got) = run(&after_a_waterfall()) else {
        return;
    };
    assert_eq!(got, vec![0; 32]);
}

/// A loop of four iterations (s7 = 0 to 3) selecting `v2 = s6 ? 1 : 0` at
/// its top. s6 starts as lanes below 16; at s7 == 1 an `if` narrows it to
/// lanes below 8 (`s_and_b32 s6, s6, s8`), and at s7 == 2 another sets it to
/// the number 0xF0. The joins after each `if` reach the loop's join only
/// over its back edge, and the second joins a mask with data: with the
/// console's lanes, each lane's bit of the number. The last iteration
/// selects by 0xF0's bits: lanes 4 to 7.
fn carried_mask() -> Vec<u32> {
    let mut body = MBCNT.to_vec();
    body.extend([
        vopc(V_CMP_GT_U32, int(16), 0), // 7: vcc = lanes < 16
        sop1(S_MOV_B32, 6, VCC_LO),     // 8
        vopc(V_CMP_GT_U32, int(8), 0),  // 9: vcc = lanes < 8
        sop1(S_MOV_B32, 8, VCC_LO),     // 10
        sop1(S_MOV_B32, 7, int(0)),     // 11: s7 = 0
        0xD501_0002,                    // 12: v_cndmask_b32 v2, 0, 1, s6
        6 << 18 | int(1) << 9 | int(0), // 13
        sopc(S_CMP_LG_U32, 7, int(1)),  // 14
        branch(S_CBRANCH_SCC1, 15, 17), // 15
        sop2(S_AND_B32, 6, 6, 8),       // 16: s6 &= s8
        sopc(S_CMP_LG_U32, 7, int(2)),  // 17
        branch(S_CBRANCH_SCC1, 18, 21), // 18
        sop1(S_MOV_B32, 6, LITERAL),    // 19: s6 = 0xF0
        0xF0,                           // 20
        sop2(S_ADD_U32, 7, 7, int(1)),  // 21: s7 += 1
        sopc(S_CMP_LT_U32, 7, int(4)),  // 22
        branch(S_CBRANCH_SCC1, 23, 12), // 23
    ]);
    program(&body)
}

/// A loop-carried mask joined with data selects by the number's bits.
#[test]
fn a_mask_carried_round_a_loop_through_joins_and_data_selects_by_its_bits() {
    let Some(got) = run(&carried_mask()) else {
        return;
    };
    let want: Vec<u32> = (0..32).map(|l| u32::from((4..8).contains(&l))).collect();
    assert_eq!(got, want);
}
