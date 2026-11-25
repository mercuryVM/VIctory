# Memory guard batching

The page guard protects aliases of tracked GPU memory. Alias candidates
may arrive unsorted and repeated, while the existing protection loop
groups only adjacent consecutive page numbers. It also scanned every
active write declaration for every candidate.

Normalize candidate pages into sorted unique order. For large batches
with more than four active declarations, sort nonempty write ranges
once and walk the two sorted streams with a prefix maximum end. For
small batches retain the simple scan. Never merge or remove entries in
the live writer registry: dropping one of two equal writing guards
must not allow protection while the other is still active.

All planning and protection remains under the original guard lock.
The existing VirtualProtect/VirtualQuery fallback, mapped/unmapped
handling, dirty tracking, and read-only policy remain unchanged.
No reduction in actual copied bytes or frame time is claimed. Sorting
can cost more for small unordered inputs; a Windows benchmark is
needed to quantify net benefit. Grouping also depends on already
protected pages, active writers and mapping boundaries.

Run:
    cargo test -p victory-gpu watch::page_batch --locked
    cargo test -p victory-gpu watch::tests --locked
    cargo clippy -p victory-gpu --all-targets --locked -- -D warnings
    cargo xtask policy
