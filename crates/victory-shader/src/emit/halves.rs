//! Console wave64 mesh programs on hosts whose waves are narrower.
//!
//! A mesh shader cannot declare a wave size (SM 6.6), so the host runs a
//! console wave of 64 lanes in whatever waves it forms: two 32-lane waves on
//! NVIDIA ("halves"), other sizes on WARP. Per-lane code does not care;
//! lanes are the console's by construction (`flat & 63`). What does:
//!
//! - `v_readfirstlane_b32` reads the first lane of the host wave, not of the
//!   console wave. That is the same value when every lane that runs it holds
//!   the same value, which is what the console compiler mostly uses it for
//!   (moving a value it proved uniform into an SGPR). [`varying`] finds the
//!   values that may differ between lanes, and [`Varying::uniform_at`] those
//!   that still do among the lanes that run a given read; only a first-lane
//!   read of one of those needs the console wave.
//! - Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt` of a mask,
//!   scalar compares) need the console wave's ballot.
//!
//! Both are exchanged through group-shared memory between group barriers
//! ([`super::lower`]'s `mesh_ballot` and `mesh_first_lane`), so they
//! need a point that every thread of the group reaches exactly once:
//! [`all_threads`]. Elsewhere a mask read as data stops with a reason, and
//! a first-lane read keeps the host wave's first lane (see the shortcut in
//! `lower`).

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// The registers `inst` reads as operands: its reads, without the old value
/// of a VGPR a vector instruction writes whole. Conservative: the old value
/// stays an operand for every other instruction, for VGPRs among the
/// sources, and for writes that keep part of it (accumulating forms, 16-bit
/// and SDWA results).
fn operand_reads(inst: &Inst) -> Vec<Reg> {
    let all = inst.reads.iter().map(|r| r.0);
    let Fields::Valu(v) = &inst.fields else {
        return all.collect();
    };
    let name = inst.name();
    if v.sdwa.is_some() || name.contains("mac") || name.contains("16") {
        return all.collect();
    }
    let source = |r: u8| {
        v.src.iter().any(|o| match *o {
            Operand::Vector { reg, dwords } => (reg..reg.saturating_add(dwords)).contains(&r),
            _ => false,
        })
    };
    all.filter(|&r| match r {
        Reg::V(n) => inst.write(r).is_none() || source(n),
        _ => true,
    })
    .collect()
}

/// Successors of `b` as lanes run the program.
fn lane_successors(st: &Structured, b: usize) -> Vec<usize> {
    match st.exits[b] {
        LaneExit::Jump(t) => vec![t],
        LaneExit::Branch {
            taken, not_taken, ..
        } => vec![taken, not_taken],
        LaneExit::End | LaneExit::Indirect => vec![],
    }
}

/// Blocks reachable from `from` (included) as lanes run the program.
fn reachable(st: &Structured, from: usize) -> Vec<bool> {
    let mut seen = vec![false; st.exits.len()];
    let mut stack = vec![from];
    seen[from] = true;
    while let Some(b) = stack.pop() {
        for s in lane_successors(st, b) {
            if !seen[s] {
                seen[s] = true;
                stack.push(s);
            }
        }
    }
    seen
}

/// Blocks on a cycle of the lanes' control flow.
fn in_cycle(st: &Structured) -> Vec<bool> {
    let n = st.exits.len();
    (0..n)
        .map(|b| {
            lane_successors(st, b)
                .into_iter()
                .any(|s| reachable(st, s)[b])
        })
        .collect()
}

/// Per block: every thread of the group runs it, exactly once. That holds
/// for a block on every path from the entry to an end that is on no loop
/// and after none (a loop can be left early by [`super::lower`]'s
/// iteration limit, which ends the lane).
pub(crate) fn all_threads(st: &Structured) -> Vec<bool> {
    let n = st.exits.len();
    let cycle = in_cycle(st);
    let after_loop: Vec<bool> = {
        let mut seen = vec![false; n];
        for b in (0..n).filter(|&b| cycle[b]) {
            for (s, r) in reachable(st, b).into_iter().enumerate() {
                seen[s] |= r;
            }
        }
        seen
    };
    let from_entry = reachable(st, 0);
    (0..n)
        .map(|b| {
            if !from_entry[b] || cycle[b] || after_loop[b] {
                return false;
            }
            // Every path from the entry passes `b`: with `b` removed, no
            // end is reachable.
            let mut seen = vec![false; n];
            let mut stack = vec![0];
            seen[0] = true;
            while let Some(c) = stack.pop() {
                if c == b {
                    continue;
                }
                if matches!(st.exits[c], LaneExit::End | LaneExit::Indirect) {
                    return false;
                }
                for s in lane_successors(st, c) {
                    if !seen[s] {
                        seen[s] = true;
                        stack.push(s);
                    }
                }
            }
            true
        })
        .collect()
}

/// What may differ between the lanes of one console wave (see [`varying`]).
struct Lanes<'a> {
    st: &'a Structured,
    /// Values that may differ.
    values: HashSet<Value>,
    /// Compare and carry results whose lane bit may differ beyond EXEC.
    bits: HashSet<Value>,
}

impl Lanes<'_> {
    fn atom_varies(&self, a: Atom) -> bool {
        match a {
            Atom::Uniform(_) => false,
            Atom::Lane(v) => self.bits.contains(&v),
            Atom::Phi(v) => self.values.contains(&v) || self.bits.contains(&v),
            Atom::Quad(_) | Atom::Opaque(_) | Atom::Constant(_) => true,
        }
    }

    /// Whether lanes of one wave that enter block `b` (and pass `guard`)
    /// can disagree on `bit`.
    fn bool_varies(&self, bit: Bool, b: usize, guard: Option<Bool>) -> bool {
        let within: Vec<Bool> = [self.st.reach[b]].into_iter().chain(guard).collect();
        self.st.bools.varies(bit, &within, &|a| self.atom_varies(a))
    }

    /// A lane mask's bit, by any of its dwords' values.
    fn mask_bit(&self, v: Value) -> Option<Bool> {
        let lo = self.st.highs.get(&v).copied().unwrap_or(v);
        self.st.masks.get(&lo).copied()
    }

    /// A branch some lanes of one console wave may take and others not.
    fn divergent(&self, b: usize) -> Option<(usize, usize)> {
        match self.st.exits[b] {
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } if taken != not_taken && self.bool_varies(when, b, None) => Some((taken, not_taken)),
            _ => None,
        }
    }
}

/// Values that may differ between the lanes of one console wave that hold
/// them. Conservative: what this does not list is the same in every lane
/// of the console wave that runs the instruction reading it.
///
/// - VGPRs on entry vary, except the mesh group in v5; SGPRs (user data,
///   the merged wave info) do not.
/// - A result varies if an operand does, or it counts lanes (`v_mbcnt`),
///   or it comes from LDS (lanes write it differently), or it reads a lane
///   mask lane by lane (a `v_cndmask_b32` selector, a carry) whose bit can
///   differ between the lanes that run it: a bit depends on compares of
///   varying data, on masks the structure cannot see into, or on phis of
///   masks that vary.
/// - A VGPR write in lanes with EXEC clear keeps the old value: where the
///   structure guards the instruction, the old value is an operand, and
///   the result varies if the guard can differ between lanes.
/// - A phi varies where lanes of one wave can arrive by different edges
///   (after a divergent branch) with different values, and everything a loop
///   left by a divergent branch defines varies.
/// - `v_readfirstlane_b32` and `v_readlane_b32` give every lane one value;
///   `v_writelane_b32` writes one lane.
pub(crate) fn varying<'a>(p: &'a Program, st: &'a Structured) -> Varying<'a> {
    let n = p.blocks.len();
    let mut l = Lanes {
        st,
        values: p
            .inputs
            .iter()
            // v5 holds the mesh group, the same in every lane
            // (`io::raise` checks it is read only by `v_readfirstlane_b32`).
            .filter(|(r, _)| matches!(r, Reg::V(n) if *n != 5))
            .map(|&(_, v)| v)
            .collect(),
        bits: HashSet::new(),
    };
    let masks: HashSet<Value> = st.masks.keys().chain(st.highs.keys()).copied().collect();
    let cycle = in_cycle(st);
    let mut divergent_loop;
    loop {
        let before = (l.values.len(), l.bits.len());
        // Blocks reachable from each side of a divergent branch.
        let splits: Vec<(Vec<bool>, Vec<bool>)> = (0..n)
            .filter_map(|b| l.divergent(b))
            .map(|(t, f)| (reachable(st, t), reachable(st, f)))
            .collect();
        // Loops some lanes leave before others: a block on a cycle with a
        // divergent branch, and every block on a cycle through it.
        divergent_loop = vec![false; n];
        for b in (0..n).filter(|&b| cycle[b] && l.divergent(b).is_some()) {
            let from = reachable(st, b);
            for c in 0..n {
                if from[c] && reachable(st, c)[b] {
                    divergent_loop[c] = true;
                }
            }
        }
        // Blocks lanes of one wave can enter by different edges.
        let joins: Vec<bool> = (0..n)
            .map(|b| {
                let preds = &p.blocks[b].preds;
                splits.iter().any(|(t, f)| {
                    (0..preds.len())
                        .any(|i| t[preds[i]] && (0..preds.len()).any(|j| i != j && f[preds[j]]))
                })
            })
            .collect();
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                let first = phi.args.first().copied();
                let differ = phi.args.iter().any(|&a| Some(a) != first);
                if divergent_loop[b]
                    || phi.args.iter().any(|a| l.values.contains(a))
                    || differ && joins[b]
                {
                    l.values.insert(phi.value);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let name = inst.name();
                if name == "v_readfirstlane_b32" || name == "v_readlane_b32" {
                    continue;
                }
                let vector = matches!(inst.fields, Fields::Valu(_));
                let guard = st.guards[b].get(i).copied().flatten();
                let written: Vec<Reg> = inst.writes.iter().map(|w| w.0).collect();
                let explicit = operand_reads(inst);
                let varies = divergent_loop[b]
                    || name.starts_with("v_mbcnt")
                    || name == "v_writelane_b32"
                    || matches!(inst.fields, Fields::Ds { .. })
                    || guard.is_some_and(|g| l.bool_varies(g, b, None))
                    || inst.reads.iter().any(|&(r, v)| {
                        if matches!(r, Reg::S(EXEC) | Reg::S(127)) {
                            return false;
                        }
                        // The old value of a VGPR written where every lane
                        // that runs it has its EXEC bit set is not read.
                        if matches!(r, Reg::V(_))
                            && written.contains(&r)
                            && guard.is_none()
                            && !explicit.contains(&r)
                        {
                            return false;
                        }
                        if l.values.contains(&v) {
                            return true;
                        }
                        vector
                            && matches!(r, Reg::S(_))
                            && l.mask_bit(v).is_some_and(|m| l.bool_varies(m, b, guard))
                    });
                if varies {
                    for &(r, w) in &inst.writes {
                        // A VALU writes masks (compares, carries) as numbers
                        // that are one per wave; their lane bits are what
                        // varies.
                        if vector && matches!(r, Reg::S(_)) && masks.contains(&w) {
                            l.bits.insert(w);
                            continue;
                        }
                        l.values.insert(w);
                    }
                }
            }
        }
        if (l.values.len(), l.bits.len()) == before {
            let mut defs = HashMap::new();
            for (b, block) in p.blocks.iter().enumerate() {
                for (k, phi) in block.phis.iter().enumerate() {
                    defs.insert(phi.value, Def::Phi(b, k));
                }
                for (i, inst) in block.insts.iter().enumerate() {
                    for w in &inst.writes {
                        defs.insert(w.1, Def::Inst(b, i));
                    }
                }
            }
            return Varying {
                p,
                lanes: l,
                divergent_loop,
                defs,
            };
        }
    }
}

#[derive(Clone, Copy)]
enum Def {
    Phi(usize, usize),
    Inst(usize, usize),
}

/// [`varying`]'s result.
pub(crate) struct Varying<'a> {
    p: &'a Program,
    lanes: Lanes<'a>,
    divergent_loop: Vec<bool>,
    defs: HashMap<Value, Def>,
}

/// Steps [`Varying::uniform_at`] follows back before it gives up.
const MAX_STEPS: usize = 4096;

impl Varying<'_> {
    /// Whether `v` may differ between lanes of one console wave somewhere.
    pub(crate) fn contains(&self, v: &Value) -> bool {
        self.lanes.values.contains(v)
    }

    /// Whether `v` is the same in every lane of a console wave that runs
    /// instruction `i` of block `b`: a value [`varying`] lists can still be,
    /// when only lanes of one side of the branches that make it differ get
    /// there. A phi takes the values of the edges such lanes can arrive by;
    /// an instruction, its operands for those lanes. The lanes are those
    /// that satisfy what holds where `i` runs (the block's reach, the
    /// instruction's guard), as the structure's booleans say.
    pub(crate) fn uniform_at(&self, v: Value, b: usize, i: usize) -> bool {
        let st = self.lanes.st;
        let mut within: Vec<(Bool, bool)> = vec![(st.reach[b], true)];
        if let Some(g) = st.guards[b].get(i).copied().flatten() {
            within.push((g, true));
        }
        let mut seen = HashSet::new();
        let mut steps = 0;
        self.uniform_in(v, &within, &mut seen, &mut steps)
    }

    fn uniform_in(
        &self,
        v: Value,
        within: &[(Bool, bool)],
        seen: &mut HashSet<Value>,
        steps: &mut usize,
    ) -> bool {
        if !self.lanes.values.contains(&v) || !seen.insert(v) {
            // (A value met again on the way is checked where first met.)
            return true;
        }
        *steps += 1;
        if *steps > MAX_STEPS {
            return false;
        }
        self.uniform_def(v, within, seen, steps)
    }

    fn uniform_def(
        &self,
        v: Value,
        within: &[(Bool, bool)],
        seen: &mut HashSet<Value>,
        steps: &mut usize,
    ) -> bool {
        let st = self.lanes.st;
        let some_lane = |extra: &[(Bool, bool)]| {
            let all: Vec<(Bool, bool)> = within.iter().chain(extra).copied().collect();
            st.bools.meet(&all)
        };
        match self.defs.get(&v) {
            None => false,
            Some(&Def::Phi(j, k)) => {
                if self.divergent_loop[j] {
                    return false;
                }
                let block = &self.p.blocks[j];
                let phi = &block.phis[k];
                // The incoming values lanes of interest can bring.
                let mut args: Vec<Value> = Vec::new();
                for (q, &arg) in block.preds.iter().zip(&phi.args) {
                    let mut edge = vec![(st.reach[*q], true)];
                    if let LaneExit::Branch {
                        when,
                        taken,
                        not_taken,
                    } = st.exits[*q]
                        && taken != not_taken
                    {
                        edge.push((when, taken == j));
                    }
                    if some_lane(&edge) && !args.contains(&arg) {
                        args.push(arg);
                    }
                }
                match args[..] {
                    [] => true,
                    [one] => self.uniform_in(one, within, seen, steps),
                    _ => false,
                }
            }
            Some(&Def::Inst(b, i)) => {
                let inst = &self.p.blocks[b].insts[i];
                let name = inst.name();
                if self.divergent_loop[b]
                    || name.starts_with("v_mbcnt")
                    || name == "v_writelane_b32"
                    || matches!(inst.fields, Fields::Ds { .. })
                {
                    return false;
                }
                let vector = matches!(inst.fields, Fields::Valu(_));
                // Lanes of interest with their EXEC bit clear keep the old
                // value of what it writes.
                let guard = st.guards[b].get(i).copied().flatten();
                let partial = guard.is_some_and(|g| some_lane(&[(g, false)]));
                if partial {
                    return false;
                }
                let written: Vec<Reg> = inst.writes.iter().map(|w| w.0).collect();
                let explicit = operand_reads(inst);
                let within_here: Vec<Bool> = within
                    .iter()
                    .filter(|w| w.1)
                    .map(|w| w.0)
                    .chain([st.reach[b]])
                    .chain(guard)
                    .collect();
                inst.reads.iter().all(|&(r, x)| {
                    if matches!(r, Reg::S(EXEC) | Reg::S(127)) {
                        return true;
                    }
                    if matches!(r, Reg::V(_)) && written.contains(&r) && !explicit.contains(&r) {
                        return true;
                    }
                    if vector
                        && matches!(r, Reg::S(_))
                        && let Some(m) = self.lanes.mask_bit(x)
                    {
                        return !st
                            .bools
                            .varies(m, &within_here, &|a| self.lanes.atom_varies(a));
                    }
                    self.uniform_in(x, within, seen, steps)
                })
            }
        }
    }
}
