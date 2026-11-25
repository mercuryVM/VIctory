//! GPU timestamps for DRED notes. Allocate another batch only when a list fills
//! the preceding one. Each batch has space for a timestamp at list close.

use super::{Note, host};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

const QUERIES: u32 = 4096;
const NOTES: usize = QUERIES as usize - 1;

struct Batch {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
}

impl Batch {
    fn new() -> Option<Self> {
        let device = &host()?.device;
        let mut heap = None;
        let mut back = None;
        let readback = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: u64::from(QUERIES) * 8,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: plain object creation, with one u64 of readback per query.
        unsafe {
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: QUERIES,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .ok()?;
            device
                .CreateCommittedResource(
                    &readback,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut back,
                )
                .ok()?;
        }
        Some(Self {
            heap: heap?,
            back: back?,
        })
    }
}

#[derive(Default)]
pub(super) struct Timestamps {
    batches: Vec<Batch>,
    allocation_failed: bool,
}

/// How many notes were timestamped in each used batch; the last batch also
/// contains the closing timestamp. A reused list may own extra unused batches.
fn layout(notes: usize, batches: usize) -> impl Iterator<Item = (u32, bool)> {
    let recorded = notes.min(batches.saturating_mul(NOTES));
    let used = recorded.div_ceil(NOTES);
    (0..used).map(move |i| ((recorded - i * NOTES).min(NOTES) as u32, i + 1 == used))
}

impl Timestamps {
    pub(super) fn record(&mut self, list: &ID3D12GraphicsCommandList, note: usize) {
        let index = note / NOTES;
        if index == self.batches.len() && !self.allocation_failed {
            if let Some(batch) = Batch::new() {
                self.batches.push(batch);
            } else {
                self.allocation_failed = true;
                static WARN: std::sync::Once = std::sync::Once::new();
                WARN.call_once(|| {
                    tracing::warn!(
                        note,
                        "GPU timing could not allocate another timestamp batch"
                    )
                });
            }
        }
        if let Some(batch) = self.batches.get(index) {
            // SAFETY: the list is open. The query is within this batch.
            unsafe {
                list.EndQuery(
                    &batch.heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    (note % NOTES) as u32,
                )
            };
        }
    }

    pub(super) fn close(&self, list: &ID3D12GraphicsCommandList, notes: usize) {
        let used = || self.batches.iter().zip(layout(notes, self.batches.len()));
        if let Some((batch, (count, _))) = used().last() {
            // SAFETY: the last batch reserves a slot for the end of the list.
            // Take its timestamp before any of the diagnostic resolves.
            unsafe { list.EndQuery(&batch.heap, D3D12_QUERY_TYPE_TIMESTAMP, count) };
        }
        for (batch, (count, last)) in used() {
            // SAFETY: these queries were recorded in the open list.
            unsafe {
                list.ResolveQueryData(
                    &batch.heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    0,
                    count + u32::from(last),
                    &batch.back,
                    0,
                );
            }
        }
    }

    pub(super) fn slowest(&self, notes: &[Note], frequency: u64, n: usize) -> Vec<(f64, String)> {
        let mut stamps = Vec::new();
        for (batch, (count, last)) in self
            .batches
            .iter()
            .zip(layout(notes.len(), self.batches.len()))
        {
            let len = count as usize + usize::from(last);
            let mut ptr = std::ptr::null_mut();
            let range = D3D12_RANGE {
                Begin: 0,
                End: len * 8,
            };
            // SAFETY: the submission finished. Close resolved these u64 values.
            unsafe {
                if batch.back.Map(0, Some(&range), Some(&mut ptr)).is_err() {
                    return Vec::new();
                }
                stamps.extend_from_slice(std::slice::from_raw_parts(ptr.cast::<u64>(), len));
                batch.back.Unmap(0, None);
            }
        }
        rank(notes, &stamps, frequency, n)
    }
}

fn rank(notes: &[Note], stamps: &[u64], frequency: u64, n: usize) -> Vec<(f64, String)> {
    let mut times: Vec<_> = stamps
        .windows(2)
        .enumerate()
        .map(|(i, pair)| (i, pair[1].saturating_sub(pair[0])))
        .collect();
    times.sort_by_key(|&(_, ticks)| std::cmp::Reverse(ticks));
    times
        .into_iter()
        .take(n)
        .map(|(i, ticks)| {
            let text = if i + 2 == stamps.len() && i + 1 < notes.len() {
                format!(
                    "aggregate of notes #{i}..#{} ({} commands, timestamp allocation incomplete)",
                    notes.len() - 1,
                    notes.len() - i
                )
            } else {
                format!(
                    "{} (note #{i})",
                    notes[i].0.chars().take(300).collect::<String>()
                )
            };
            (ticks as f64 * 1e3 / frequency.max(1) as f64, text)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_cover_only_recorded_queries_and_a_single_end() {
        let batches = |notes, capacity| layout(notes, capacity).collect::<Vec<_>>();
        assert!(batches(0, 2).is_empty());
        assert!(batches(10, 0).is_empty());
        assert_eq!(batches(NOTES, 2), [(4095, true)]);
        assert_eq!(batches(NOTES + 1, 2), [(4095, false), (1, true)]);
        assert_eq!(batches(6719, 3), [(4095, false), (2624, true)]);
        assert_eq!(batches(6719, 1), [(4095, true)]);
    }

    #[test]
    fn list_tail_retains_correct_attribution_with_and_without_a_second_batch() {
        let notes = vec![("dispatch".to_string(), None); 6719];
        let mut stamps: Vec<u64> = (0..=6719).collect();
        for t in &mut stamps[4095..] {
            *t += 100;
        }
        assert_eq!(
            rank(&notes, &stamps, 1000, 1),
            [(101., "dispatch (note #4094)".to_owned())]
        );
        let mut partial: Vec<u64> = (0..4095).collect();
        partial.push(*stamps.last().unwrap());
        let worst = rank(&notes, &partial, 1000, 1);
        assert_eq!(worst[0].0, 2725.);
        assert!(worst[0].1.contains("#4094..#6718 (2625 commands"));
    }
}
