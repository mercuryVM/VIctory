//! Repository maintenance. Run with `cargo xtask <command>`.

mod policy;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enforce the asset policy: the repository holds only small text files
    /// we wrote. Checks tracked and untracked-but-not-ignored files.
    Policy {
        /// Check only what is staged for commit (used by the pre-commit hook).
        #[arg(long)]
        staged: bool,
    },
    /// Run quality assurance gates: asset policy, code formatting, and workspace lints.
    Qa {
        /// Automatically format files with cargo fmt instead of just checking.
        #[arg(long)]
        fix: bool,
    },
    /// Point git at .githooks/ so every commit runs the policy check.
    InstallHooks,
}

fn main() -> Result<()> {
    let root = workspace_root();
    match Cli::parse().command {
        Cmd::Policy { staged } => policy::run(&root, staged),
        Cmd::Qa { fix } => run_qa(&root, fix),
        Cmd::InstallHooks => {
            let status = Command::new("git")
                .args(["config", "core.hooksPath", ".githooks"])
                .current_dir(&root)
                .status()
                .context("running git")?;
            anyhow::ensure!(
                status.success(),
                "git config failed (is this a git repository?)"
            );
            println!("hooks installed: commits now run `cargo xtask policy --staged`");
            Ok(())
        }
    }
}

fn run_qa(root: &Path, fix: bool) -> Result<()> {
    println!("==> Step 1/3: Checking asset policy (Rule #1)...");
    policy::run(root, false)?;

    println!("==> Step 2/3: Checking code formatting...");
    let fmt_status = if fix {
        Command::new("cargo")
            .args(["fmt", "--all"])
            .current_dir(root)
            .status()
            .context("running cargo fmt")?
    } else {
        Command::new("cargo")
            .args(["fmt", "--all", "--", "--check"])
            .current_dir(root)
            .status()
            .context("running cargo fmt --check")?
    };
    if !fmt_status.success() {
        bail!("code formatting check failed; run `cargo fmt` or `cargo xtask qa --fix`");
    }

    println!("==> Step 3/3: Running workspace lints (clippy)...");
    let clippy_status = Command::new("cargo")
        .args(["clippy", "--workspace", "--all-targets"])
        .current_dir(root)
        .status()
        .context("running cargo clippy")?;
    if !clippy_status.success() {
        bail!("clippy lint check failed");
    }

    println!("All quality assurance checks passed: policy, format, and lints OK.");
    Ok(())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace")
        .into()
}
