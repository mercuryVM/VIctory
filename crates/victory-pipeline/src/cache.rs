use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{Context, Result};

use crate::Converter;

/// Identifies one conversion result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey(blake3::Hash);

impl CacheKey {
    pub fn new(converter: &dyn Converter, input: &[u8]) -> Self {
        let id = converter.id();
        let mut h = blake3::Hasher::new();
        // Length-prefix the id so a NUL inside it cannot be mistaken for the
        // old separator and collide with a different id/version pair.
        h.update(&(id.len() as u64).to_le_bytes());
        h.update(id.as_bytes());
        h.update(&converter.version().to_le_bytes());
        h.update(input);
        CacheKey(h.finalize())
    }

    pub fn to_hex(&self) -> String {
        self.0.to_hex().to_string()
    }
}

enum InFlightState {
    Pending,
    Done(Result<Vec<u8>, String>),
}

struct InFlight {
    state: Mutex<InFlightState>,
    done: Condvar,
}

impl InFlight {
    fn new() -> Self {
        Self {
            state: Mutex::new(InFlightState::Pending),
            done: Condvar::new(),
        }
    }

    fn wait(&self) -> Result<Vec<u8>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match &*state {
                InFlightState::Pending => {
                    state = self.done.wait(state).unwrap_or_else(|e| e.into_inner());
                }
                InFlightState::Done(Ok(bytes)) => return Ok(bytes.clone()),
                InFlightState::Done(Err(msg)) => return Err(anyhow::anyhow!("{msg}")),
            }
        }
    }

    fn finish(&self, result: Result<Vec<u8>, String>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        *state = InFlightState::Done(result);
        self.done.notify_all();
    }
}

/// Ensures an in-flight entry is always removed and waiters are woken even if
/// the converting thread returns early or panics.
struct InFlightGuard {
    map: Arc<Mutex<HashMap<CacheKey, Arc<InFlight>>>>,
    key: CacheKey,
    slot: Arc<InFlight>,
    completed: bool,
}

impl InFlightGuard {
    /// Removes `key` from the in-flight map and publishes `bytes` only if
    /// another thread is actually waiting on `slot` (avoiding an extra clone in
    /// the uncontended case).
    fn complete_ok(mut self, bytes: Vec<u8>) -> Vec<u8> {
        self.completed = true;
        self.remove_from_map();
        if Arc::strong_count(&self.slot) > 1 {
            self.slot.finish(Ok(bytes.clone()));
        }
        bytes
    }

    fn complete_err(mut self, msg: String) {
        self.completed = true;
        self.remove_from_map();
        if Arc::strong_count(&self.slot) > 1 {
            self.slot.finish(Err(msg));
        }
    }

    fn remove_from_map(&self) {
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        if map
            .get(&self.key)
            .is_some_and(|s| Arc::ptr_eq(s, &self.slot))
        {
            map.remove(&self.key);
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.remove_from_map();
            self.slot
                .finish(Err("cache conversion aborted before completion".to_owned()));
        }
    }
}

/// Content-addressed store under the user's cache directory.
#[derive(Clone)]
pub struct Cache {
    root: PathBuf,
    in_flight: Arc<Mutex<HashMap<CacheKey, Arc<InFlight>>>>,
}

impl std::fmt::Debug for Cache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cache").field("root", &self.root).finish()
    }
}

impl Cache {
    /// Opens (creating) a cache at `root`. Refuses locations inside the repo.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        victory_core::ensure_outside_repo(&root)?;
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        Ok(Cache {
            root,
            in_flight: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn path_for(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.root.join(&hex[..2]).join(&hex[2..])
    }

    /// Returns the cached output path for `input`, converting it first if needed.
    /// Concurrent callers for the same key deduplicate and wait for one conversion.
    pub fn get_or_convert(&self, converter: &dyn Converter, input: &[u8]) -> Result<PathBuf> {
        let key = CacheKey::new(converter, input);
        let path = self.path_for(&key);
        if path.is_file() {
            tracing::trace!(converter = converter.id(), key = %key.to_hex(), "cache hit");
            return Ok(path);
        }
        let (path, _, _) = self.get_or_convert_bytes(converter, input)?;
        Ok(path)
    }

    /// Returns `(path, bytes, from_cache)` for `input`.
    ///
    /// On a miss, returns the newly converted bytes directly in memory without
    /// re-reading the file back from disk. Concurrent callers requesting the
    /// same key deduplicate onto a single `converter.convert` invocation and
    /// receive the bytes in memory when it finishes.
    pub fn get_or_convert_bytes(
        &self,
        converter: &dyn Converter,
        input: &[u8],
    ) -> Result<(PathBuf, Vec<u8>, bool)> {
        let key = CacheKey::new(converter, input);
        let path = self.path_for(&key);
        if let Ok(bytes) = std::fs::read(&path) {
            tracing::trace!(converter = converter.id(), key = %key.to_hex(), "cache hit");
            return Ok((path, bytes, true));
        }

        let guard = {
            let mut map = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(slot) = map.get(&key).cloned() {
                drop(map);
                let bytes = slot.wait()?;
                return Ok((path, bytes, true));
            }
            let slot = Arc::new(InFlight::new());
            map.insert(key, slot.clone());
            InFlightGuard {
                map: self.in_flight.clone(),
                key,
                slot,
                completed: false,
            }
        };

        // Re-check disk after registering in-flight to close the window where a
        // previous leader finished between our fast-path read and map lookup.
        if let Ok(bytes) = std::fs::read(&path) {
            let bytes = guard.complete_ok(bytes);
            return Ok((path, bytes, true));
        }

        let output = match converter
            .convert(input)
            .with_context(|| format!("converter {} v{}", converter.id(), converter.version()))
        {
            Ok(output) => output,
            Err(err) => {
                guard.complete_err(format!("{err:#}"));
                return Err(err);
            }
        };
        if let Err(err) = write_atomic(&path, &output) {
            guard.complete_err(format!("{err:#}"));
            return Err(err);
        }
        let output = guard.complete_ok(output);
        Ok((path, output, false))
    }

    /// Stores precomputed `output` bytes under `(converter, input)` if not
    /// already present on disk. Returns `(path, written)`.
    pub fn put(
        &self,
        converter: &dyn Converter,
        input: &[u8],
        output: &[u8],
    ) -> Result<(PathBuf, bool)> {
        let key = CacheKey::new(converter, input);
        let path = self.path_for(&key);
        if path.is_file() {
            return Ok((path, false));
        }
        write_atomic(&path, output)?;
        Ok((path, true))
    }
}

/// Writes `bytes` to a unique sibling temp file and renames it into `path`,
/// removing the staging file if either write or rename fails.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().expect("hashed path has a parent");
    std::fs::create_dir_all(dir)?;
    let tmp = tempfile_in(dir);
    if let Err(err) = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err.into());
    }
    Ok(())
}

fn tempfile_in(dir: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".tmp-{}-{n}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Upper {
        version: u32,
        runs: AtomicUsize,
    }

    impl Converter for Upper {
        fn id(&self) -> &'static str {
            "test.upper"
        }
        fn version(&self) -> u32 {
            self.version
        }
        fn convert(&self, input: &[u8]) -> Result<Vec<u8>> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            Ok(input.to_ascii_uppercase())
        }
    }

    #[test]
    fn converts_once_then_hits() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let a = cache.get_or_convert(&conv, b"abc").unwrap();
        let b = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!(a, b);
        assert_eq!(std::fs::read(&a).unwrap(), b"ABC");
        assert_eq!(conv.runs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failed_publication_removes_the_staging_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let path = cache.path_for(&CacheKey::new(&conv, b"abc"));
        std::fs::create_dir_all(&path).unwrap();

        assert!(cache.get_or_convert(&conv, b"abc").is_err());
        assert!(path.is_dir());
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
    }

    #[test]
    fn version_bump_invalidates() {
        let v1 = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let v2 = Upper {
            version: 2,
            runs: AtomicUsize::new(0),
        };
        assert_ne!(CacheKey::new(&v1, b"x"), CacheKey::new(&v2, b"x"));
    }

    struct Keyed(&'static str, u32);

    impl Converter for Keyed {
        fn id(&self) -> &'static str {
            self.0
        }
        fn version(&self) -> u32 {
            self.1
        }
        fn convert(&self, _: &[u8]) -> Result<Vec<u8>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn id_and_version_are_delimited() {
        assert_ne!(
            CacheKey::new(&Keyed("a\0", 1), b""),
            CacheKey::new(&Keyed("a", 256), b"")
        );
        assert_ne!(
            CacheKey::new(&Keyed("a", 1), b"b"),
            CacheKey::new(&Keyed("a", 1), b"")
        );
    }

    #[test]
    fn refuses_cache_inside_repo() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        assert!(Cache::open(inside).is_err());
    }

    #[test]
    fn get_or_convert_bytes_returns_direct_bytes_without_reread() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let (p1, b1, hit1) = cache.get_or_convert_bytes(&conv, b"hello").unwrap();
        assert!(!hit1, "first lookup converts");
        assert_eq!(b1, b"HELLO");
        assert_eq!(conv.runs.load(Ordering::Relaxed), 1);

        let (p2, b2, hit2) = cache.get_or_convert_bytes(&conv, b"hello").unwrap();
        assert!(hit2, "second lookup hits disk");
        assert_eq!(p1, p2);
        assert_eq!(b2, b"HELLO");
        assert_eq!(conv.runs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn concurrent_same_key_deduplicates_to_single_conversion() {
        use std::sync::Barrier;
        use std::time::Duration;

        struct SlowUpper {
            runs: AtomicUsize,
        }

        impl Converter for SlowUpper {
            fn id(&self) -> &'static str {
                "test.slow_upper"
            }
            fn version(&self) -> u32 {
                1
            }
            fn convert(&self, input: &[u8]) -> Result<Vec<u8>> {
                self.runs.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(50));
                Ok(input.to_ascii_uppercase())
            }
        }

        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = SlowUpper {
            runs: AtomicUsize::new(0),
        };
        let threads = 8;
        let barrier = Barrier::new(threads);

        std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let cache = cache.clone();
                    let conv = &conv;
                    let barrier = &barrier;
                    s.spawn(move || {
                        barrier.wait();
                        cache.get_or_convert_bytes(conv, b"concurrent-key").unwrap()
                    })
                })
                .collect();

            let mut misses = 0;
            for h in handles {
                let (_, bytes, hit) = h.join().unwrap();
                assert_eq!(bytes, b"CONCURRENT-KEY");
                if !hit {
                    misses += 1;
                }
            }
            assert_eq!(misses, 1, "only the leader reports a miss");
        });
        assert_eq!(
            conv.runs.load(Ordering::SeqCst),
            1,
            "converter ran exactly once across all concurrent callers"
        );
    }

    #[test]
    fn staging_temp_file_is_cleaned_up_on_rename_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let key = CacheKey::new(&conv, b"fail-rename");
        let target = cache.path_for(&key);
        // Place a directory at the destination path so `rename` fails.
        std::fs::create_dir_all(&target).unwrap();

        assert!(cache.get_or_convert_bytes(&conv, b"fail-rename").is_err());

        let parent = target.parent().unwrap();
        let leftover_tmps: Vec<_> = std::fs::read_dir(parent)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with(".tmp-"))
            .collect();
        assert!(
            leftover_tmps.is_empty(),
            "staging temp files leaked: {leftover_tmps:?}"
        );
    }

    #[test]
    fn put_writes_once_and_hits_thereafter() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let (p1, written1) = cache.put(&conv, b"warm", b"WARM").unwrap();
        assert!(written1);
        let (p2, written2) = cache.put(&conv, b"warm", b"WARM").unwrap();
        assert!(!written2);
        assert_eq!(p1, p2);
        let (_, bytes, hit) = cache.get_or_convert_bytes(&conv, b"warm").unwrap();
        assert!(hit);
        assert_eq!(bytes, b"WARM");
        assert_eq!(conv.runs.load(Ordering::Relaxed), 0);
    }
}
