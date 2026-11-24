//! Pixel programs that read their barycentrics as data, not only through
//! `v_interp_*`: the i/j pair `SPI_PS_INPUT_ADDR` enables becomes y and z
//! of `SV_Barycentrics`, interpolated as the pair is. Synthetic GFX10
//! programs, checked in the container, validated by `dxil.dll` when
//! present, and drawn on the host GPU and WARP when they support
//! barycentrics.

use super::container::SigElement;
use super::{Container, FourCC, HostShader, emit};
use crate::native::{HwStage, NativeProgram, PsInput, RootParam, RootType, StageIo, Usage};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const PERSP_CENTER: u32 = 1 << 1;
const PERSP_CENTROID: u32 = 1 << 2;
const LINEAR_CENTER: u32 = 1 << 5;

/// The weights of vertices 0, 1 and 2 to target 0: 1 - i - j, i, j (the
/// pair is in v0 and v1), and 1.
const WEIGHTS: [u32; 6] = [
    0x0804_00F2, // v_sub_f32 v2, 1.0, v0
    0x0804_0302, // v_sub_f32 v2, v2, v1
    0x7E06_02F2, // v_mov_b32 v3, 1.0
    0xF800_180F,
    0x0301_0002, // exp mrt0 v2, v0, v1, v3 done vm
    S_ENDPGM,
];

/// A float signature row of `cols` channels (`row` none: no register).
fn element(name: &str, kind: u8, row: Option<u32>, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp: 0,
        row,
        col: 0,
        cols,
        usage: 0,
        half: false,
    }
}

/// Translates pixel program `words`, its VGPRs as `input_addr` enables
/// them, with `isg1` as the console's input signature (`attrs` its stage
/// data), user data in `layout`, and one float4 target.
fn pixel(
    words: &[u32],
    input_addr: u32,
    isg1: &[SigElement],
    attrs: Vec<PsInput>,
    layout: Vec<RootParam>,
) -> Result<HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout,
        used: vec![],
        io: StageIo::Ps {
            inputs: attrs,
            input_addr,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let isg1 = super::container::signature(isg1, true);
    let target = element("SV_Target", 16, Some(0), 4);
    let osg1 = super::container::signature(&[target], false);
    let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&bytes)?;
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let io = io::raise(&p, &native)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let ps = emit(&c, &native, &p, &st, &io, &res, super::Target::ANY)?;
    validate(&ps.dxil);
    Ok(ps)
}

/// Validates with `dxil.dll` when there is one; our hash must be its.
#[cfg(windows)]
fn validate(dxil: &[u8]) {
    use super::validator::Validator;
    let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) else {
        eprintln!("no dxil.dll; not validated");
        return;
    };
    let mut copy = dxil.to_vec();
    v.validate(&mut copy).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(copy[4..20], dxil[4..20], "container hash");
}

#[cfg(not(windows))]
fn validate(_: &[u8]) {}

/// An `ISG1` row: system value, semantic index, register, component type,
/// mask, channels read.
type Row = (u32, u32, u32, u32, u8, u8);

/// What a translation declares: its `ISG1` rows, its `PSV0` input
/// elements (kind, interpolation mode; read only when no input has a
/// register, so that the elements end the part) and `SFI0`.
fn declared(dxil: &[u8]) -> (Vec<Row>, Vec<(u8, u8)>, u64) {
    let c = Container::parse(dxil).unwrap();
    let data = |f| c.part(f).unwrap().data;
    let u = |d: &[u8], at: usize| u32::from_le_bytes(d[at..at + 4].try_into().unwrap());
    let isg1 = data(FourCC::ISG1);
    let rows: Vec<Row> = (0..u(isg1, 0) as usize)
        .map(|i| {
            let at = u(isg1, 4) as usize + 32 * i;
            let r = |k| u(isg1, at + k);
            (r(12), r(8), r(20), r(16), isg1[at + 24], isg1[at + 25])
        })
        .collect();
    let psv = data(FourCC::PSV0);
    let (ins, outs) = (usize::from(psv[4 + 28]), usize::from(psv[4 + 29]));
    let at = psv.len() - 16 * (ins + outs);
    let modes = if rows.iter().all(|r| r.2 == u32::MAX) {
        (0..ins).map(|k| (psv[at + 16 * k + 11], psv[at + 16 * k + 13])).collect()
    } else {
        Vec::new()
    };
    let features = u64::from_le_bytes(data(FourCC::SFI0).try_into().unwrap());
    (rows, modes, features)
}

/// `SV_Barycentrics` `index`: no register, float3, y and z read.
const fn barycentrics(index: u32) -> Row {
    (23, index, u32::MAX, 3, 0b111, 0b110)
}

/// The `SFI0` bit of barycentrics.
const BARYCENTRICS: u64 = 0x2_0000;

#[test]
fn barycentrics_read_as_data_are_sv_barycentrics_y_and_z() {
    // Each pair (`SPI_PS_INPUT_ADDR` bit: perspective sample, center,
    // centroid, linear sample, center, centroid) and the mode it should get.
    let pairs = [(1 << 0, 6), (1 << 1, 2), (1 << 2, 3), (1 << 4, 7), (1 << 5, 4), (1 << 6, 5)];
    let none: &[SigElement] = &[];
    let mut cases: Vec<_> = pairs.iter().map(|&(a, m)| (a, m, none)).collect();
    // The row DXC writes for `float3 b : SV_Barycentrics` gives way to the
    // pair read (a signature has no mode).
    let console = [element("SV_Barycentrics", 28, None, 3)];
    cases.push((LINEAR_CENTER, 4, &console));
    for (addr, mode, isg1) in cases {
        let ps = pixel(&WEIGHTS, addr, isg1, vec![], vec![])
            .unwrap_or_else(|e| panic!("{addr:#x}: {e}"));
        let (rows, modes, features) = declared(&ps.dxil);
        assert_eq!(rows, [barycentrics(0)], "{addr:#x}");
        assert_eq!(modes, [(28, mode)], "{addr:#x}");
        assert_ne!(features & BARYCENTRICS, 0, "{addr:#x}");
    }
}

#[test]
fn a_perspective_and_a_linear_pair_are_two_elements() {
    // exp mrt0 v0, v1, v2, v3 done vm: i and j of two pairs.
    let both = [0xF800_180F, 0x0302_0100, S_ENDPGM];
    let ps = pixel(&both, PERSP_CENTER | LINEAR_CENTER, &[], vec![], vec![]).unwrap();
    let (rows, modes, _) = declared(&ps.dxil);
    assert_eq!(rows, [barycentrics(0), barycentrics(1)]);
    assert_eq!(modes, [(28, 2), (28, 4)]);
    // D3D has one perspective-correct SV_Barycentrics, not center and
    // centroid both.
    let e = pixel(&both, PERSP_CENTER | PERSP_CENTROID, &[], vec![], vec![]).unwrap_err();
    assert!(e.to_string().contains("two sample positions"), "{e}");
}

#[test]
fn interpolation_alone_declares_no_barycentrics() {
    let words = [
        0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
        0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
        0xF800_180F,
        0x0202_0202, // exp mrt0 v2, v2, v2, v2 done vm
        S_ENDPGM,
    ];
    let attr = PsInput {
        usage: Usage::Generic,
        row: 1,
        mask: 3,
        attr: 0,
        flat: false,
        fp16: false,
        pack2: false,
        row2: 1,
    };
    let texcoord = element("TEXCOORD", 0, Some(1), 2);
    let ps = pixel(&words, PERSP_CENTER, &[texcoord], vec![attr], vec![]).unwrap();
    let (rows, _, features) = declared(&ps.dxil);
    assert_eq!(rows, [(0, 0, 1, 3, 0b11, 0b01)]);
    assert_eq!(features & BARYCENTRICS, 0);
}

#[test]
fn a_barycentric_joined_at_a_branch_is_loaded_for_the_join() {
    // i, or 0.5 when root constant 0 is not 0: a phi of the input.
    let mut words = vec![
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 -> v_sub_f32
        0x7E00_02F0, // v_mov_b32 v0, 0.5
    ];
    words.extend(WEIGHTS);
    let layout = vec![RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    let ps = pixel(&words, PERSP_CENTER, &[], vec![], layout).unwrap();
    assert_eq!(declared(&ps.dxil).0, [barycentrics(0)]);
}

/// A vertex shader passing POSITION (float4) on as `SV_Position`.
#[cfg(windows)]
fn passthrough_vs() -> HostShader {
    use super::bitcode::{Attr, Term};
    let mut s = super::shader::Shader::new(false);
    let mut input = element("POSITION", 0, Some(0), 4);
    input.usage = 15;
    let mut output = element("SV_Position", 3, Some(0), 4);
    (output.usage, output.interp) = (15, 4);
    s.inputs.push(input);
    s.outputs.push(output);
    let b = s.entry;
    for c in 0..4 {
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(c), s.m.undef(s.t.i32)];
        let x = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(c), x];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    s.m.terminate(b, Term::Ret);
    HostShader {
        stage: HwStage::Vs,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
        group_waves: 1,
    }
}

#[cfg(windows)]
fn barycentrics_supported(host: &super::host::Host) -> bool {
    use windows::Win32::Graphics::Direct3D12::*;
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
    // SAFETY: a feature query with its own out structure.
    let ok = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS3,
            (&raw mut o).cast(),
            size_of_val(&o) as u32,
        )
    };
    ok.is_ok() && o.BarycentricsSupported.as_bool()
}

/// A triangle whose vertices have w 1, 2 and 4, drawn with the weights
/// program: every pixel well inside holds the weights D3D gives vertices
/// 0, 1 and 2 there, perspective-correct for the perspective pair and
/// screen-linear for the linear one (they differ by up to about 0.3).
#[test]
#[cfg(windows)]
fn drawn_weights_are_the_hosts_vertex_weights() {
    use super::host::{Draw, Host, Root};
    let ndc = [[-0.8f32, -0.7], [0.9, -0.6], [-0.5, 0.85]];
    let w = [1.0f32, 2.0, 4.0];
    let vertices: Vec<f32> = (0..3)
        .flat_map(|k| [ndc[k][0] * w[k], ndc[k][1] * w[k], 0.5 * w[k], w[k]])
        .collect();
    let (width, height) = (32u32, 32u32);
    // Screen-linear and perspective-correct weights at a pixel's centre.
    let weights = |px: u32, py: u32| {
        let p = [
            (px as f32 + 0.5) / width as f32 * 2.0 - 1.0,
            1.0 - (py as f32 + 0.5) / height as f32 * 2.0,
        ];
        let cross = |a: [f32; 2], b: [f32; 2]| a[0] * b[1] - a[1] * b[0];
        let sub = |a: [f32; 2], b: [f32; 2]| [a[0] - b[0], a[1] - b[1]];
        let (e1, e2, d) = (sub(ndc[1], ndc[0]), sub(ndc[2], ndc[0]), sub(p, ndc[0]));
        let (l1, l2) = (cross(d, e2) / cross(e1, e2), cross(e1, d) / cross(e1, e2));
        let linear = [1.0 - l1 - l2, l1, l2];
        let q = [linear[0] / w[0], linear[1] / w[1], linear[2] / w[2]];
        let s = q[0] + q[1] + q[2];
        (linear, [q[0] / s, q[1] / s, q[2] / s])
    };
    let vs = passthrough_vs();
    validate(&vs.dxil);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        if !barycentrics_supported(&host) {
            eprintln!("{label}: no barycentrics support; skipped");
            continue;
        }
        for (addr, linear) in [(PERSP_CENTER, false), (LINEAR_CENTER, true)] {
            let ps = pixel(&WEIGHTS, addr, &[], vec![], vec![]).unwrap();
            let (root, pso) = host
                .pipeline(&vs, &ps)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let draw = Draw {
                roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(vec![0]))],
                textures: &[],
                samplers: &[],
                vertices: &vertices,
                stride: 16,
                size: (width, height),
                mesh_groups: 0,
            };
            let texels = host
                .draw(&root, &pso, &draw)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let (mut inside, mut spread) = (0, 0.0f32);
            for py in 0..height {
                for px in 0..width {
                    let (lin, persp) = weights(px, py);
                    // Near or past an edge: coverage is the rasteriser's.
                    if lin.iter().any(|&l| l < 0.02) {
                        continue;
                    }
                    inside += 1;
                    spread = (0..3).fold(spread, |m, c| m.max((lin[c] - persp[c]).abs()));
                    let want = if linear { lin } else { persp };
                    let t = texels[(py * width + px) as usize];
                    let ok = (0..3).all(|c| (t[c] - want[c]).abs() < 2e-3) && t[3] == 1.0;
                    assert!(ok, "{label} {addr:#x} ({px}, {py}): {t:?}, want {want:?}");
                }
            }
            assert!(inside > 100 && spread > 0.2, "{inside} pixels, {spread}");
            eprintln!("{label} {addr:#x}: {inside} pixels match");
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no device with barycentrics; nothing drawn");
    }
}
