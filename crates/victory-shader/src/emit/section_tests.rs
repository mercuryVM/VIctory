//! Whole-wave sections whose cross-lane steps run under a partial EXEC:
//! hand-assembled GFX10 compute programs (wave32) lowered to DXIL,
//! validated by `dxil.dll` when present, and run on the host GPU.

use super::host::{Dispatch, Host, Root, View};
use super::lanemask_tests::lower_cs;

/// Lanes 0 to 19 of 32 (an `if`) take the exclusive prefix sum of
/// `lane + 1`, as the console compiler expands `WavePrefixSum`: `row_shr`
/// steps, then lane 15's row total added to the second row by
/// `v_permlanex16` with EXEC on that row only and op_sel FI (fetch the
/// inactive lanes), and the lane's own term taken off. Each running lane
/// stores its prefix at dword `lane`.
const PREFIX: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0 (the table)
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0A02,
    0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    0x4A06_0081, // v_add_nc_u32 v3, 1, v0
    0x7D88_0094, // v_cmp_gt_u32 vcc_lo, 20, v0
    0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
    0xBE94_3DC1, // s_or_saveexec_b32 s20, -1
    0xD501_0002,
    0x0052_0680, // v_cndmask_b32_e64 v2, 0, v3, s20
    0x7E02_0302, // v_mov_b32 v1, v2
    0x4A02_02FA,
    0xFF09_1101, // v_add_nc_u32_dpp v1, v1, v1 row_shr:1 bound_ctrl:1
    0x4A02_02FA,
    0xFF01_1201, // v_add_nc_u32_dpp v1, v1, v1 row_shr:2
    0x4A02_02FA,
    0xFF01_1401, // v_add_nc_u32_dpp v1, v1, v1 row_shr:4
    0x4A02_02FA,
    0xFF01_1801, // v_add_nc_u32_dpp v1, v1, v1 row_shr:8
    0xBEFE_03FF,
    0xFFFF_0000, // s_mov_b32 exec_lo, 0xffff0000
    0xD778_0804,
    0x0305_8301, // v_permlanex16_b32 v4, v1, -1, -1 op_sel:[1,0]
    0x4A02_0304, // v_add_nc_u32_e32 v1, v4, v1
    0xBEFE_03C1, // s_mov_b32 exec_lo, -1
    0x4C02_0501, // v_sub_nc_u32_e32 v1, v1, v2
    0xBEFE_0314, // s_mov_b32 exec_lo, s20
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE070_1000,
    0x800A_0110, // buffer_store_dword v1, v16, s[40:43], 0 offen
    0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
    0xBF81_0000, // s_endpgm
];

/// The prefix lowers (and validates) without a GPU.
#[test]
fn a_prefix_fetching_inactive_lanes_lowers() {
    lower_cs(PREFIX).expect("lowers");
}

#[test]
fn a_prefix_fetching_inactive_lanes_sums_the_lanes_below() {
    let shader = lower_cs(PREFIX).expect("lowers");
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
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
    // Lane l (below 20) holds 1 + 2 + ... + l.
    let want: Vec<u32> = (0..32u32)
        .map(|l| if l < 20 { l * (l + 1) / 2 } else { 0 })
        .collect();
    assert_eq!(words[0], want);
}
