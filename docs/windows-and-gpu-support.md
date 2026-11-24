# Windows and GPU support

Status snapshot for `main` at `ffe310e` (2026-11-24). The pull requests
linked below are experimental and are not included in that revision; check
their status before relying on them.

## Windows 10

VIctory needs Direct3D shader model 6.6. The built-in D3D12 runtime on Windows
10 may stop at shader model 6.5, below that requirement, as reported in
[#1612](https://github.com/Marvisak/VIctory/issues/1612). Agility SDK support
is being explored in [#175](https://github.com/Marvisak/VIctory/pull/175) and
[#969](https://github.com/Marvisak/VIctory/pull/969); neither is part of
`main` in this snapshot.

## GPUs without mesh shaders

The world renderer on `main` requires mesh shader support, so a GPU without
it cannot render the world. The compute fallback in
[#961](https://github.com/Marvisak/VIctory/pull/961) is still an open PR, not
a supported `main` feature.

Some shader workloads also need 64-lane waves. The open
[#189](https://github.com/Marvisak/VIctory/pull/189) explores Wave64
emulation on narrower host waves; results may depend on the GPU and driver.

## Before reporting a run

Include the GPU model and driver, Windows build, VIctory commit, any PRs on
top of `main`, and the stage where the run stopped. The
[README](../README.md#getting-started) lists the system requirements and
explains the Windows commit error `1455`. Keep game files and game-derived
data out of issues and pull requests; see the [asset policy](asset-policy.md).

For the detailed, changing status and hardware reports, see
[issue #1612](https://github.com/Marvisak/VIctory/issues/1612).
