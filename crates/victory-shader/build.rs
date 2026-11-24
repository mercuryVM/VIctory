use std::path::{Path, PathBuf};

fn main() {
    source_hash();
    agility_runtime();
}

/// `VICTORY_SHADER_SOURCE_HASH`: a BLAKE3 digest of this crate's `src/**/*.rs`, so the
/// shader cache key changes when the translator does and survives unrelated rebuilds.
fn source_hash() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by Cargo"),
    );
    let src_dir = manifest_dir.join("src");
    let mut files = Vec::new();
    collect_rs(&src_dir, &src_dir, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut hasher = blake3::Hasher::new();
    for (rel, full) in files {
        println!("cargo:rerun-if-changed={}", full.display());
        hasher.update(rel.as_bytes());
        hasher.update(&[0]);
        let contents = std::fs::read(&full)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", full.display()));
        hasher.update(&(contents.len() as u64).to_le_bytes());
        hasher.update(&contents);
    }
    println!(
        "cargo:rustc-env=VICTORY_SHADER_SOURCE_HASH={}",
        hasher.finalize().to_hex()
    );
}

fn collect_rs(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("failed to read directory {}: {e}", dir.display()))
        .map(|entry| entry.expect("directory entry"))
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let rel = path
                .strip_prefix(root)
                .expect("path under root")
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, path));
        }
    }
}

/// Opts the test binaries into an app-local Agility SDK runtime when one sits in
/// `target/<profile>/D3D12/`.
fn agility_runtime() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    // `OUT_DIR` is `target/<profile>/build/<pkg>-<hash>/out`; the profile directory is three
    // levels up. The application looks for the runtime in `target/<profile>/D3D12/`, so this
    // crate reads it from the same place instead of inventing a second convention.
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let profile = out
        .ancestors()
        .nth(3)
        .expect("a profile directory above OUT_DIR")
        .to_path_buf();
    let runtime = profile.join("D3D12").join("D3D12Core.dll");
    println!("cargo:rerun-if-changed={}", runtime.display());
    if !runtime.is_file() {
        // No app-local runtime: leave the in-box one alone, exactly as a build without
        // the opt-in would.
        return;
    }

    // The loader resolves `D3D12SDKPath` next to the running module, and test binaries run
    // from `deps`, so mirror the runtime there before the tests need it.
    let deps = profile.join("deps").join("D3D12");
    let _ = std::fs::create_dir_all(&deps);
    let _ = std::fs::copy(&runtime, deps.join("D3D12Core.dll"));

    // Export the two symbols the Agility SDK loader looks for in the host module, so
    // d3d12.dll loads that runtime instead of the in-box copy. Without them Windows 10
    // reports shader model 6.5 and every SM 6.6 test is skipped.
    println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKVersion,DATA");
    println!("cargo:rustc-link-arg=/EXPORT:D3D12SDKPath,DATA");
}
