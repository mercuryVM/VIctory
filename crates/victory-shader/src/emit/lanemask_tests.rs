//! Lane masks read as data where they hold lanes that do not run where
//! they are written: hand-assembled GFX10 compute programs (wave32, 32
//! lanes) lowered to DXIL, validated by `dxil.dll` when present, and run on
//! the host GPU against the mask computed here.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

/// `s3 = (lanes < 16 and even) | (lanes >= 24)`, made inside an `if` that
/// only lanes below 16 run, so it holds lanes 24 to 31, which do not run
/// there. With `after`, it is read as data (`v_mov_b32 v2, s3`) once EXEC
/// is back to every lane, else still inside the `if`. Each lane stores v2
/// at dword `lane` of the raw UAV at table entry 0.
fn program(after: bool) -> Vec<u32> {
    let mut w = vec![
        0xBE84_0300, // s_mov_b32 s4, s0 (the table)
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xD4C4_0006,
        0x0001_2F00, // v_cmp_gt_u32_e64 s6, v0, 23 (lanes >= 24)
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0 (lanes < 16)
        0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
        0x3602_0081, // v_and_b32 v1, 1, v0
        0xD4C2_0002,
        0x0002_0280, // v_cmp_eq_u32_e64 s2, 0, v1 (even lanes)
        0x8803_0602, // s_or_b32 s3, s2, s6
    ];
    let read = 0x7E04_0203; // v_mov_b32 v2, s3
    if after {
        w.push(0x887E_0A7E); // s_or_b32 exec_lo, exec_lo, s10
        w.push(read);
    } else {
        w.push(read);
        w.push(0x887E_0A7E); // s_or_b32 exec_lo, exec_lo, s10
    }
    w.extend([
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x800A_0210, // buffer_store_dword v2, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

/// [`program`]'s mask read as data by a lane-local step inside a whole-wave
/// section (`s_or_saveexec_b32 s20, -1` ... `s_mov_b32 exec_lo, s20`)
/// within the `if`: EXEC there is every lane, but only lanes 0 to 15 run
/// the section on the host, so lanes 24 to 31's bits are not seen.
fn in_section() -> Vec<u32> {
    let mut w = program(false);
    // Replace the read inside the `if` by a section around it.
    let read = w.iter().position(|&x| x == 0x7E04_0203).expect("the read");
    w.splice(
        read..=read,
        [
            0xBE94_3DC1, // s_or_saveexec_b32 s20, -1
            0x4A04_0003, // v_add_nc_u32 v2, s3, v0 (lane-local)
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
        ],
    );
    w
}

/// The mask the program builds.
const MASK: u32 = 0x5555 | 0xFF00_0000;

/// Lowers a wave32 compute program of 32 threads with a table holding one
/// raw UAV at s0.
fn lower_cs(words: &[u32]) -> Result<HostShader, ShaderError> {
    lower_cs_on(words, false)
}

/// [`lower_cs`] in wave64 (`wave64`, 64 threads) or wave32 (32 threads).
fn lower_cs_on(words: &[u32], wave64: bool) -> Result<HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
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
            threads: [if wave64 { 64 } else { 32 }, 1, 1],
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
        target: super::Target::ANY,
        console_cf: false,
        per_lane_only: false,
    })?;
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
    }
    Ok(shader)
}

/// Read as data once every lane runs again: each lane stores the whole
/// mask, the lanes that did not run where it was made included.
#[test]
fn a_mask_holding_lanes_elsewhere_reads_whole_where_they_all_run() {
    let shader = lower_cs(&program(true)).expect("lowers");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // A one-wave program reading lanes: the console's wave size (WaveSize
    // 32), which WARP's pipeline creation refuses; hardware only.
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
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
    assert_eq!(words[0], vec![MASK; 32], "each lane stores the whole mask");
}

/// Read as data where lanes 24 to 31 do not run: no ballot there sees
/// them, so the translation still refuses.
#[test]
fn a_mask_holding_lanes_elsewhere_is_refused_where_they_do_not_run() {
    match lower_cs(&program(false)) {
        Ok(_) => panic!("lowered a mask read where some of its lanes do not run"),
        Err(e) => assert!(
            format!("{e:?}").contains("a lane mask holding lanes elsewhere read as data"),
            "{e:?}"
        ),
    }
}

/// A wave64 program writes the low half of EXEC with a constant
/// (`s_mov_b32 exec_lo, 0xffff0000`: lanes 0 to 15 stop, 16 to 63 run) and
/// reads it as a number (`v_add_nc_u32 v2, exec_lo, v0`), then restores
/// EXEC. Each running lane stores v2 at dword `lane`.
const CONSTANT_MASK: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0 (the table)
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    0xBE8A_047E, // s_mov_b64 s[10:11], exec
    0xBEFE_03FF,
    0xFFFF_0000, // s_mov_b32 exec_lo, 0xffff0000
    0x4A04_007E, // v_add_nc_u32 v2, exec_lo, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000,
    0x800A_0210, // buffer_store_dword v2, v16, s[40:43], 0 offen
    0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    0xBF81_0000, // s_endpgm
];

/// EXEC's low half, written with a constant, is that number as data.
#[test]
fn a_constant_exec_half_reads_as_its_constant() {
    let shader = lower_cs_on(CONSTANT_MASK, true).expect("lowers");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            // The console's wave size (64), which a 32-lane host refuses.
            eprintln!("{label}: pipeline refused (wave size); skipped");
            continue;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..64u32)
            .map(|l| if l < 16 { 0 } else { 0xFFFF_0000 + l })
            .collect();
        assert_eq!(words[0], want, "{label}");
    }
}

/// Inside a whole-wave section only the lanes active on entry run on the
/// host, whatever EXEC says: the mask is still refused there.
#[test]
fn a_mask_holding_lanes_elsewhere_is_refused_inside_a_whole_wave_section() {
    match lower_cs(&in_section()) {
        Ok(_) => panic!("lowered a mask read inside a section where some of its lanes do not run"),
        Err(e) => assert!(
            format!("{e:?}").contains("a lane mask holding lanes elsewhere read as data"),
            "{e:?}"
        ),
    }
}
