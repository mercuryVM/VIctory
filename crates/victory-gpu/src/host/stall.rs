//! A bounded snapshot of the submitted group's CPU-side records while its
//! fence is pending. Timestamp readback and DRED are unavailable at this point.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use super::{HostList, INFINITE};

pub(super) fn timeout(already_reported: bool) -> u32 {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !already_reported && *ENABLED.get_or_init(|| crate::switch::on("VICTORY_GPU_STALL_REPORT")) {
        3_000
    } else {
        INFINITE
    }
}

fn group_range(group: usize, per_group: usize, runs: usize) -> std::ops::Range<usize> {
    let start = group * per_group;
    start..start.saturating_add(per_group).min(runs)
}

/// Direct dimensions count host groups. An indirect record count does not
/// reveal dispatch dimensions before execution.
fn dispatch(note: &str) -> Option<(&str, Option<u64>)> {
    let (command, tail) = note.split_once(" [compute ")?;
    let (hash, _) = tail.split_once(']')?;
    if let Some(dimensions) = command.strip_prefix("dispatch ") {
        let (x, yz) = dimensions.split_once('x')?;
        let (y, z) = yz.split_once('x')?;
        let groups = x
            .parse::<u64>()
            .ok()?
            .saturating_mul(y.parse().ok()?)
            .saturating_mul(z.parse().ok()?);
        return Some((hash, Some(groups)));
    }
    command
        .starts_with("indirect kind 2 ")
        .then_some((hash, None))
}

pub(super) fn report(
    fence: u64,
    group: usize,
    group_count: usize,
    runs: &[&HostList],
    per_group: usize,
) {
    let range = group_range(group, per_group, runs.len());
    let lists = &runs[range.clone()];
    let list_ids: Vec<_> = lists.iter().take(8).map(|list| list.id).collect();
    let mut pipelines = BTreeSet::new();
    let mut calls = BTreeMap::<String, (usize, usize)>::new();
    let mut largest = None;
    let mut command_count = 0;
    for list in lists {
        let bound = list.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        pipelines.extend(bound.iter().cloned());
        drop(bound);
        let notes = list.notes.lock().unwrap_or_else(|e| e.into_inner());
        command_count += notes.len();
        for (ordinal, (note, _)) in notes.iter().enumerate() {
            let Some((hash, groups)) = dispatch(note) else {
                continue;
            };
            let count = calls.entry(hash.to_owned()).or_default();
            if let Some(groups) = groups {
                count.0 += 1;
                if largest
                    .as_ref()
                    .is_none_or(|(previous, _)| groups > *previous)
                {
                    let header = note.split(" | ").next().unwrap_or(note);
                    largest = Some((groups, format!("list {} #{ordinal}: {header}", list.id)));
                }
            } else {
                count.1 += 1;
            }
        }
    }
    let mut frequent: Vec<_> = calls.into_iter().collect();
    frequent.sort_by_key(|(_, (direct, indirect))| std::cmp::Reverse(direct + indirect));
    frequent.truncate(6);
    let pipeline_count = pipelines.len();
    let pipeline_sample: Vec<_> = pipelines.into_iter().take(10).collect();
    tracing::warn!(
        fence, group = group + 1, group_count, first_run = range.start, runs = lists.len(),
        ?list_ids, command_count, pipeline_count, ?pipeline_sample,
        compute_direct_indirect = ?frequent, largest_direct = ?largest,
        "host fence pending after 3 s; recorded work in the submitted group"
    );
}

#[test]
fn timed_singleton_and_short_last_batch_name_their_own_runs() {
    assert_eq!(group_range(3, 1, 5), 3..4);
    assert_eq!(group_range(2, 2, 5), 4..5);
    assert_eq!(group_range(0, usize::MAX, 5), 0..5);
    assert_eq!(
        dispatch("dispatch 9x3x2 [compute 0123] | bindings"),
        Some(("0123", Some(54)))
    );
    assert_eq!(
        dispatch("indirect kind 2 max 1 at 0xaa [compute abcd]"),
        Some(("abcd", None))
    );
    assert_eq!(
        dispatch("DROPPED dispatch (no pipeline) [compute abcd]"),
        None
    );
}
