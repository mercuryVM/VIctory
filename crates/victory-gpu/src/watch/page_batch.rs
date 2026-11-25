//! Plan protection candidates without changing any page protection.
//! The caller holds the guard lock throughout planning and application.

/// Sorted, distinct pages not covered by an active write declaration.
/// Busy ranges are half-open; duplicate and overlapping declarations
/// remain independent in the caller's registry.
pub(super) fn eligible(pages: &[usize], busy: &[(usize, usize)]) -> Vec<usize> {
    let mut pages = pages.to_vec();
    if !pages.is_sorted() {
        pages.sort_unstable();
    }
    pages.dedup();
    if pages.is_empty() || busy.is_empty() {
        return pages;
    }
    // Avoid constructing an index for small batches or few writers.
    if busy.len() <= 4 || pages.len() <= 4 {
        pages.retain(|p| !busy.iter().any(|&(a, b)| (a..b).contains(p)));
        return pages;
    }

    let mut spans: Vec<_> = busy.iter().copied().filter(|&(a, b)| a < b).collect();
    spans.sort_unstable();
    // Starts and candidate pages are sorted. Consume every declaration
    // whose start is no later than the current page, maintaining the
    // maximum end, including for nested and duplicate declarations.
    let mut i = 0;
    let mut end = 0;
    pages.retain(|&p| {
        while i < spans.len() && spans[i].0 <= p {
            end = end.max(spans[i].1);
            i += 1;
        }
        p >= end
    });
    pages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(pages: &[usize], busy: &[(usize, usize)]) -> Vec<usize> {
        let mut out: Vec<_> = pages
            .iter()
            .copied()
            .filter(|p| !busy.iter().any(|&(a, b)| (a..b).contains(p)))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    #[test]
    fn aliases_are_sorted_and_deduplicated() {
        assert_eq!(eligible(&[8, 2, 1, 2, 3, 8, 7], &[]), [1, 2, 3, 7, 8]);
        assert!(eligible(&[], &[(0, usize::MAX)]).is_empty());
    }

    #[test]
    fn nested_duplicate_and_empty_writes_are_excluded() {
        let pages: Vec<_> = (0..20).rev().collect();
        let busy = [(8, 9), (1, 12), (1, 12), (6, 7), (14, 14), (17, 19), (16, 15)];
        assert_eq!(eligible(&pages, &busy), [0, 12, 13, 14, 15, 16, 19]);
        assert_eq!(eligible(&[usize::MAX, usize::MAX - 1], &[(usize::MAX - 2, usize::MAX)]), [usize::MAX]);
    }

    #[test]
    fn dropping_one_duplicate_writer_does_not_enable_its_pages() {
        let pages = [0, 1, 2, 3, 4, 5, 6];
        let mut busy = vec![(1, 5), (1, 5), (2, 6), (0, 0), (3, 3), (9, 10)];
        assert_eq!(eligible(&pages, &busy), [0, 6]);
        busy.swap_remove(0);
        assert_eq!(eligible(&pages, &busy), [0, 6]);
        busy.swap_remove(1);
        assert_eq!(eligible(&pages, &busy), [0, 1, 6]);
    }

    #[test]
    fn exhaustive_small_candidates_and_active_writes() {
        let ranges: Vec<_> = (0..5).flat_map(|a| (0..5).map(move |b| (a, b))).collect();
        for code in 0..5usize.pow(4) {
            let pages = [code % 5, code / 5 % 5, code / 25 % 5, code / 125];
            for &a in &ranges {
                for &b in &ranges {
                    let busy = [a, b, a, b, (4, 4)];
                    assert_eq!(eligible(&pages, &busy), reference(&pages, &busy));
                    let large = [5, 4, 3, 2, 1, 0, 4];
                    assert_eq!(eligible(&large, &busy), reference(&large, &busy));
                }
            }
        }
    }

    #[test]
    fn seeded_large_batches_match_the_reference() {
        let mut seed = 0x7a21_3b49_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as usize
        };
        for _ in 0..2048 {
            let n = next() % 256;
            let m = next() % 64;
            let pages: Vec<_> = (0..n).map(|_| next() % 1024).collect();
            let busy: Vec<_> = (0..m)
                .map(|_| {
                    let a = next() % 1024;
                    (a, a + next() % 64)
                })
                .collect();
            assert_eq!(eligible(&pages, &busy), reference(&pages, &busy));
        }
    }
}
