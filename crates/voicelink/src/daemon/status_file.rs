//! The status file: a [`DaemonStatus`] snapshot written atomically (temporary
//! file in the same directory, then rename), plus latency bookkeeping.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::status::{BackendNames, DaemonStatus, LinkState};

/// Writes `status` to `path` atomically.
pub(crate) fn write_atomic(path: &Path, status: &DaemonStatus) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "status.json".into());
    let tmp = dir.join(format!(".{name}.tmp-{}", std::process::id()));
    let result = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&serde_json::to_vec_pretty(status).map_err(std::io::Error::other)?)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

struct Inner {
    path: Option<PathBuf>,
    status: Mutex<DaemonStatus>,
    transcript_at: Mutex<Option<Instant>>,
}

/// Shared, thread-safe owner of the status snapshot.
#[derive(Clone)]
pub(crate) struct StatusHandle {
    inner: Arc<Inner>,
}

impl StatusHandle {
    /// Creates the handle in the `Starting` state; writes to `path` when given.
    pub(crate) fn new(path: Option<PathBuf>, backends: BackendNames) -> Self {
        let status = DaemonStatus {
            pid: std::process::id(),
            updated_ms: now_ms(),
            state: LinkState::Starting,
            focus: None,
            muted: false,
            backends,
            last_stt_ms: None,
            last_reply_ms: None,
            last_error: None,
        };
        let h = StatusHandle {
            inner: Arc::new(Inner {
                path,
                status: Mutex::new(status),
                transcript_at: Mutex::new(None),
            }),
        };
        h.update(|_| {});
        h
    }

    fn lock(&self) -> MutexGuard<'_, DaemonStatus> {
        self.inner.status.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Mutates the snapshot, stamps it and persists it.
    pub(crate) fn update(&self, f: impl FnOnce(&mut DaemonStatus)) {
        let mut s = self.lock();
        f(&mut s);
        s.updated_ms = now_ms();
        if let Some(path) = &self.inner.path {
            if let Err(e) = write_atomic(path, &s) {
                tracing::warn!("cannot write status file {}: {e}", path.display());
            }
        }
    }

    /// A copy of the current snapshot.
    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> DaemonStatus {
        self.lock().clone()
    }

    /// Records the instant a transcript became available (start of the reply clock).
    pub(crate) fn mark_transcript(&self) {
        *self
            .inner
            .transcript_at
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
    }

    /// Records the first audio of a reply; stores transcript-to-audio latency.
    pub(crate) fn mark_first_audio(&self) {
        let started = self
            .inner
            .transcript_at
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(t) = started {
            let ms = t.elapsed().as_millis().min(u128::from(u32::MAX)) as u32;
            self.update(|s| s.last_reply_ms = Some(ms));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> BackendNames {
        BackendNames {
            stt: "s".into(),
            llm: "l".into(),
            tts: "t".into(),
        }
    }

    #[test]
    fn write_is_atomic_and_leaves_no_temporary_file() {
        let dir = std::env::temp_dir().join(format!("vl-status-{}", std::process::id()));
        let path = dir.join("nested").join("status.json");
        let h = StatusHandle::new(Some(path.clone()), names());
        h.update(|s| {
            s.state = LinkState::Listening;
            s.last_stt_ms = Some(120);
        });
        let read: DaemonStatus = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(read.state, LinkState::Listening);
        assert_eq!(read.last_stt_ms, Some(120));
        assert_eq!(read.pid, std::process::id());
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reply_latency_runs_from_transcript_to_first_audio_once() {
        let h = StatusHandle::new(None, names());
        h.mark_first_audio(); // no transcript yet: nothing recorded
        assert_eq!(h.snapshot().last_reply_ms, None);
        h.mark_transcript();
        std::thread::sleep(std::time::Duration::from_millis(15));
        h.mark_first_audio();
        let first = h.snapshot().last_reply_ms.unwrap();
        assert!(first >= 15);
        h.mark_first_audio(); // second chunk of the same reply does not move it
        assert_eq!(h.snapshot().last_reply_ms, Some(first));
    }
}
