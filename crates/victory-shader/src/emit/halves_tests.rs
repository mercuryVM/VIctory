//! Console wave64 mesh programs run as host waves of any size (see
//! [`super::halves`]): hand-assembled GFX10 mesh programs of two console
//! waves (128 threads) whose lanes read the first lane and the ballot of
//! their console wave. Each thread stores what it got to a UAV, which the
//! test compares with what RDNA2 gives by the ISA: `v_readfirstlane_b32`
//! reads the lowest lane with its EXEC bit set (lane 0 when there is none),
//! `s_bcnt1_i32_b64` and `s_ff1_i32_b64` count and find the set bits of
//! all 64 lanes, and `v_mbcnt_*` counts the set bits below the lane. On an
//! NVIDIA GPU each console wave runs as two 32-lane waves, so a first lane
//! or a bit in the upper half crosses host waves; WARP forms other waves.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    HwStage, MeshIo, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const VCC: u8 = 106;
const EXEC: u8 = 126;
/// Inline constants: integers 0..64 and -1.
const fn k(n: u8) -> u8 {
    128 + n
}
const MINUS_ONE: u16 = 193;

/// SOP1 `op sdst, src0`.
const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SOPP `op simm16`.
const fn sopp(op: u8, imm: u16) -> u32 {
    0xBF80_0000 | (op as u32) << 16 | imm as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0`.
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..255 scalar or constant, 256.. VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOPC `op vcc, src0, vsrc1`.
const fn vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2`.
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen
/// offset:byte_off`.
const fn buffer_store_offen(vdata: u8, vaddr: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

const fn v(n: u8) -> u16 {
    256 + n as u16
}

const V_MOV_B32: u16 = 0x01;
const V_READFIRSTLANE_B32: u16 = 0x02;
const V_LSHLREV_B32: u8 = 0x1A;
const V_ADD_NC_U32: u8 = 0x25;
const V_CMP_GT_U32: u8 = 0xC4;
const V_CMP_LE_U32: u8 = 0xC3;
const V_MBCNT_LO: u16 = 0x365;
const V_MBCNT_HI: u16 = 0x366;
const S_AND_SAVEEXEC_B64: u8 = 0x24;
const S_BCNT1_I32_B64: u8 = 0x10;
const S_FF1_I32_B64: u8 = 0x14;
const S_OR_B64: u8 = 0x11;
const S_AND_B32: u8 = 0x0E;
const S_LSHR_B32: u8 = 0x20;
const S_CBRANCH_EXECZ: u8 = 0x08;

/// Threads per group: two console waves.
const THREADS: u32 = 128;
/// Dwords each thread stores, one block of `THREADS` per result.
const RESULTS: u32 = 3;

/// The prologue every program here starts with: the root constants arrive
/// in s8..s11 (mesh user data follows 8 system SGPRs), the view table
/// address in s12; the UAV's descriptor goes to s[16:19]. Then v1 = the
/// lane (`v_mbcnt` of all lanes), s4 = the wave in the group (bits 24-27 of
/// the merged wave info in s3), v2 = the thread, v3 = its first result's
/// byte offset.
fn prologue() -> Vec<u32> {
    let mut w = vec![sop1(0x03, 13, k(0))]; // s_mov_b32 s13, 0
    w.extend(s_load_dwordx4(16, 12));
    w.extend(vop3(V_MBCNT_LO, 1, MINUS_ONE, u16::from(k(0)), 0));
    w.extend(vop3(V_MBCNT_HI, 1, MINUS_ONE, v(1), 0));
    w.push(sop2(S_LSHR_B32, 4, 3, k(24)));
    w.push(sop2(S_AND_B32, 4, 4, k(15)));
    w.push(vop1(V_MOV_B32, 2, 4)); // v_mov_b32 v2, s4
    w.push(vop2(V_LSHLREV_B32, 2, u16::from(k(6)), 2));
    w.push(vop2(V_ADD_NC_U32, 2, v(2), 1));
    w.push(vop2(V_LSHLREV_B32, 3, u16::from(k(2)), 2));
    w
}

/// The end of every program: each lane exports a position at the origin
/// (D3D12 wants one from a mesh shader feeding a pixel shader; no primitive
/// is exported, so nothing is drawn).
fn end() -> Vec<u32> {
    let mut w: Vec<u32> = (8..12)
        .map(|r| vop1(V_MOV_B32, r, u16::from(k(0))))
        .collect();
    // exp pos0 v8, v9, v10, v11 done
    w.extend([0xF800_08CF, 0x0B0A_0908]);
    w.push(S_ENDPGM);
    w
}

/// Lowers a wave64 mesh program of [`THREADS`] threads, no outputs and no
/// LDS, with root constants (4 dwords) and one raw UAV.
fn build(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: true,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Ms(MeshIo {
            outputs: vec![],
            clip_cntl: 0,
            threads: THREADS,
            max_vertices: THREADS,
            max_primitives: THREADS,
            topology: 2,
            lds_bytes: 0,
        }),
        code,
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
        outputs: vec![super::container::SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        }],
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    let shader = HostShader {
        stage: HwStage::Ms,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        stride_table: lowered.stride_table,
    };
    if let Ok(mut val) = validator::Validator::load(validator::Validator::default_path().as_deref())
    {
        val.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    Ok(shader)
}

/// Runs `shader` on the hardware adapter and WARP once per root-constant
/// set of `cases` and calls `check(adapter, case, thread, results)`.
fn run(
    shader: &HostShader,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, [u32; 4], u32, [u32; 3]),
) {
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
            .mesh_pipeline(shader, &super::host::stub_ps())
            .unwrap_or_else(|e| panic!("{label}: mesh pipeline: {e}"));
        for &inputs in cases {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch_mesh(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(THREADS * RESULTS)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch: {e}"));
            for t in 0..THREADS {
                let at = |r: u32| out[0][(r * THREADS + t) as usize];
                check(label, inputs, t, [at(0), at(1), at(2)]);
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Lanes at or above s9 run `v_readfirstlane_b32` of a value that differs
/// in every thread (16 × thread + s8), under EXEC with no branch: each
/// console wave's first lane is lane s9, in the upper half from 32 on, and
/// lane 0 when s9 leaves no lane.
#[test]
fn first_lane_of_a_console_wave_crosses_host_waves() {
    let mut w = prologue();
    w.push(vop2(V_LSHLREV_B32, 6, u16::from(k(4)), 2)); // v6 = 16 * thread
    w.push(vop2(V_ADD_NC_U32, 6, 8, 6)); // + s8
    w.push(vopc(V_CMP_LE_U32, 9, 1)); // vcc = s9 <= lane
    w.push(sop1(S_AND_SAVEEXEC_B64, 6, VCC)); // s[6:7]
    w.push(vop1(V_READFIRSTLANE_B32, 5, v(6))); // s5
    w.push(sop2(S_OR_B64, EXEC, EXEC, 6));
    w.push(vop1(V_MOV_B32, 7, 5));
    w.extend(buffer_store_offen(7, 3, 16, 0));
    // And the same read with every lane on, beside it.
    w.push(vop1(V_READFIRSTLANE_B32, 5, v(6)));
    w.push(vop1(V_MOV_B32, 7, 5));
    w.extend(buffer_store_offen(7, 3, 16, (THREADS * 4) as u16));
    w.extend(end());
    let shader = build(&w).expect("lowers");
    let cases: Vec<[u32; 4]> = [0, 1, 5, 31, 32, 33, 40, 63, 64]
        .iter()
        .map(|&first| [1000, first, 0, 0])
        .collect();
    let mut bad = Vec::new();
    run(&shader, &cases, |label, [base, first, ..], t, r| {
        let wave = t / 64;
        let lane = if first < 64 { first } else { 0 };
        let want = [16 * (64 * wave + lane) + base, 16 * 64 * wave + base];
        if r[..2] != want {
            bad.push(format!(
                "{label} s9={first} thread {t}: {:?}, want {want:?}",
                &r[..2]
            ));
        }
    });
    assert!(
        bad.is_empty(),
        "{} wrong, first: {:#?}",
        bad.len(),
        &bad[..bad.len().min(8)]
    );
}

/// A lane mask over both halves of the console wave read as numbers: the
/// lanes at or above s9. `s_bcnt1_i32_b64` counts 64 - s9 of them,
/// `s_ff1_i32_b64` finds lane s9 (-1 for none), and `v_mbcnt` gives each
/// lane the set lanes below it.
#[test]
fn ballots_count_both_halves_of_a_console_wave() {
    let mut w = prologue();
    w.push(vopc(V_CMP_LE_U32, 9, 1)); // vcc = s9 <= lane
    w.push(sop1(S_BCNT1_I32_B64, 5, VCC));
    w.push(vop1(V_MOV_B32, 7, 5));
    w.extend(buffer_store_offen(7, 3, 16, 0));
    w.push(sop1(S_FF1_I32_B64, 5, VCC));
    w.push(vop1(V_MOV_B32, 7, 5));
    w.extend(buffer_store_offen(7, 3, 16, (THREADS * 4) as u16));
    w.extend(vop3(V_MBCNT_LO, 7, u16::from(VCC), u16::from(k(0)), 0));
    w.extend(vop3(V_MBCNT_HI, 7, u16::from(VCC + 1), v(7), 0));
    w.extend(buffer_store_offen(7, 3, 16, (2 * THREADS * 4) as u16));
    w.extend(end());
    let shader = build(&w).expect("lowers");
    let cases: Vec<[u32; 4]> = [0, 1, 17, 31, 32, 33, 50, 63, 64]
        .iter()
        .map(|&first| [0, first, 0, 0])
        .collect();
    let mut bad = Vec::new();
    run(&shader, &cases, |label, [_, first, ..], t, r| {
        let lane = t % 64;
        let count = 64u32.saturating_sub(first);
        let ff1 = if first < 64 { first } else { u32::MAX };
        let below = lane.saturating_sub(first);
        let want = [count, ff1, below];
        if r != want {
            bad.push(format!(
                "{label} s9={first} thread {t}: {r:?}, want {want:?}"
            ));
        }
    });
    assert!(
        bad.is_empty(),
        "{} wrong, first: {:#?}",
        bad.len(),
        &bad[..bad.len().min(8)]
    );
}

/// A first-lane read of a value every lane holds alike needs no exchange,
/// inside a branch too: lanes below s9 enter it and read s8 back.
#[test]
fn first_lane_of_uniform_data_runs_in_divergent_code() {
    let mut w = prologue();
    w.push(vop1(V_MOV_B32, 7, 8)); // v7 = s8 in every lane
    w.push(vop1(V_MOV_B32, 6, u16::from(k(0))));
    w.push(vopc(V_CMP_GT_U32, 9, 1)); // vcc = s9 > lane
    w.push(sop1(S_AND_SAVEEXEC_B64, 10, VCC)); // s[10:11]
    w.push(sopp(S_CBRANCH_EXECZ, 2));
    w.push(vop1(V_READFIRSTLANE_B32, 5, v(7)));
    w.push(vop1(V_MOV_B32, 6, 5));
    w.push(sop2(S_OR_B64, EXEC, EXEC, 10));
    w.extend(buffer_store_offen(6, 3, 16, 0));
    w.extend(end());
    let shader = build(&w).expect("lowers");
    let cases: Vec<[u32; 4]> = [0, 20, 40, 64].iter().map(|&n| [777, n, 0, 0]).collect();
    let mut bad = Vec::new();
    run(&shader, &cases, |label, [base, n, ..], t, r| {
        let want = if t % 64 < n { base } else { 0 };
        if r[0] != want {
            bad.push(format!("{label} s9={n} thread {t}: {}, want {want}", r[0]));
        }
    });
    assert!(
        bad.is_empty(),
        "{} wrong, first: {:#?}",
        bad.len(),
        &bad[..bad.len().min(8)]
    );
}

/// Where only some threads of the group get, a first-lane read of data that
/// differs between lanes cannot be exchanged: it still lowers, reading the
/// host wave's first lane (see the shortcut in `lower`).
#[test]
fn first_lane_of_varying_data_in_divergent_code_lowers() {
    let mut w = prologue();
    w.push(vopc(V_CMP_GT_U32, 9, 1)); // vcc = s9 > lane
    w.push(sop1(S_AND_SAVEEXEC_B64, 10, VCC));
    w.push(sopp(S_CBRANCH_EXECZ, 2));
    w.push(vop1(V_READFIRSTLANE_B32, 5, v(2))); // the thread: varies
    w.push(vop1(V_MOV_B32, 6, 5));
    w.push(sop2(S_OR_B64, EXEC, EXEC, 10));
    w.extend(buffer_store_offen(6, 3, 16, 0));
    w.extend(end());
    build(&w).expect("lowers");
}

/// The encodings above, as the disassembler reads them.
#[test]
fn encodings_disassemble_as_meant() {
    let text = |words: &[u32]| {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, true, 16).unwrap();
        p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .map(|i| i.machine.text(true))
            .collect::<Vec<_>>()
    };
    let mut w = prologue();
    w.push(vopc(V_CMP_LE_U32, 9, 1));
    w.push(sop1(S_AND_SAVEEXEC_B64, 6, VCC));
    w.push(sop1(S_BCNT1_I32_B64, 5, VCC));
    w.push(sop1(S_FF1_I32_B64, 5, VCC));
    w.extend(buffer_store_offen(7, 3, 16, 512));
    w.push(S_ENDPGM);
    let t = text(&w);
    eprintln!("{t:#?}");
    let want = [
        "s_mov_b32",
        "s_load_dwordx4",
        "v_mbcnt_lo_u32_b32",
        "v_mbcnt_hi_u32_b32",
        "s_lshr_b32",
        "s_and_b32",
        "v_mov_b32",
        "v_lshlrev_b32",
        "v_add_nc_u32",
        "v_lshlrev_b32",
        "v_cmp_le_u32",
        "s_and_saveexec_b64",
        "s_bcnt1_i32_b64",
        "s_ff1_i32_b64",
        "buffer_store_dword",
        "s_endpgm",
    ];
    for (got, want) in t.iter().zip(want) {
        assert!(got.starts_with(want), "{got} is not {want}");
    }
    assert!(
        t[14].contains("offen") && t[14].contains("512"),
        "{}",
        t[14]
    );
}
