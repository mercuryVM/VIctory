//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees, `external` host
//! libraries recording into title lists. Planned:
//! `present` (swap chain).

#[cfg(windows)]
mod adapter;
#[cfg(windows)]
mod bindless;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
#[cfg(windows)]
pub mod external;
pub mod fence;
pub mod frame;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
pub mod list;
#[cfg(windows)]
mod memory;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod shadow;
pub mod signature;
mod switch;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod twin;
#[cfg(windows)]
mod watch;
#[cfg(windows)]
mod inplace;
#[cfg(all(windows, test))]
mod zero_copy_probe;

#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
/// The runtime reserved title GPU memory `[base, base + len)`
/// (`XMemVirtualAlloc`): the host GPU may use it in place
/// (`VICTORY_ZERO_COPY`).
#[cfg(windows)]
pub fn title_memory_allocated(base: usize, len: usize) {
    inplace::allocated(base, len);
}

/// The title is about to free title memory at `at` (`len` bytes, or with
/// `release` the whole allocation): heaps the host GPU has open on it go
/// first (`VICTORY_ZERO_COPY`).
#[cfg(windows)]
pub fn title_memory_freeing(at: usize, len: usize, release: bool) {
    inplace::freeing(at, len, release);
}

#[cfg(windows)]
pub use watch::{
    Writing as KernelWrite, title_memory_aliased, title_memory_remapped,
    writing as kernel_will_write,
};
