//! The event spool: appending (recorder side) and tailing (controller side).
//!
//! Hooks of several panes can fire at once, so appends take an exclusive
//! `flock` on a sibling `.lock` file while they rotate and write. The tail
//! polls (cheap, and immune to inotify quirks over odd filesystems) and
//! survives rotation to `events.jsonl.1`, truncation and late creation.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use fs4::FileExt;
use voicelink_proto::{
    MAX_LINE_BYTES, SPOOL_MAX_BYTES, SPOOL_RELATIVE_PATH, SpoolRecord, decode_line, encode_line,
};

/// `$HOME` joined with the protocol's relative spool path.
pub fn default_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(PathBuf::from(home).join(SPOOL_RELATIVE_PATH))
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(suffix);
    path.with_file_name(name)
}

/// Appends one record as a single `write` on an `O_APPEND` descriptor,
/// rotating the file to `<name>.1` first when it exceeds [`SPOOL_MAX_BYTES`].
pub fn append(path: &Path, record: &SpoolRecord) -> io::Result<()> {
    append_with_limit(path, record, SPOOL_MAX_BYTES)
}

fn append_with_limit(path: &Path, record: &SpoolRecord, max_bytes: u64) -> io::Result<()> {
    let line = encode_line(record).map_err(io::Error::other)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)?;
    }
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(sibling(path, ".lock"))?;
    FileExt::lock(&lock)?;
    if fs::metadata(path).is_ok_and(|m| m.len() > max_bytes) {
        fs::rename(path, sibling(path, ".1"))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(line.as_bytes())?;
    // The lock is released when `lock` is dropped.
    Ok(())
}

/// Follows the spool from its end at construction time.
pub struct SpoolTail {
    path: PathBuf,
    file: Option<File>,
    ino: u64,
    offset: u64,
    partial: Vec<u8>,
    /// The first bytes of the file as last seen. Every record starts with a
    /// timestamp, so a changed prefix reveals a truncate-and-rewrite that the
    /// size alone would miss.
    head: Vec<u8>,
}

/// How many leading bytes identify a file generation.
const HEAD_BYTES: u64 = 48;

impl SpoolTail {
    /// Starts at the current end of the file (or at its beginning once it is
    /// created, if it does not exist yet).
    pub fn new(path: PathBuf) -> Self {
        let mut tail = SpoolTail {
            path,
            file: None,
            ino: 0,
            offset: 0,
            partial: Vec::new(),
            head: Vec::new(),
        };
        if let Ok(mut f) = File::open(&tail.path) {
            if let (Ok(meta), Ok(end)) = (f.metadata(), f.seek(SeekFrom::End(0))) {
                tail.ino = meta.ino();
                tail.offset = end;
                tail.file = Some(f);
                tail.refresh_head();
            }
        }
        tail
    }

    /// Reads whatever was appended since the last call. Malformed lines are
    /// skipped. Never blocks.
    pub fn poll(&mut self) -> Vec<SpoolRecord> {
        let mut out = Vec::new();
        if !self.head_matches() {
            self.restart_in_place();
        }
        self.drain(&mut out);
        self.refresh_head();
        match fs::metadata(&self.path) {
            Ok(meta) if self.file.is_none() || meta.ino() != self.ino => {
                // New file (first creation, or rotation): the old handle was
                // drained above, so move over and read from the start.
                if let Ok(f) = File::open(&self.path) {
                    self.ino = meta.ino();
                    self.file = Some(f);
                    self.offset = 0;
                    self.partial.clear();
                    self.head.clear();
                    self.drain(&mut out);
                    self.refresh_head();
                }
            }
            Ok(meta) if meta.len() < self.offset => {
                self.restart_in_place();
                self.drain(&mut out);
                self.refresh_head();
            }
            _ => {}
        }
        out
    }

    fn head_matches(&self) -> bool {
        use std::os::unix::fs::FileExt as _;
        let Some(file) = self.file.as_ref() else {
            return true;
        };
        if self.head.is_empty() {
            return true;
        }
        let mut buf = vec![0u8; self.head.len()];
        matches!(file.read_at(&mut buf, 0), Ok(n) if n == buf.len()) && buf == self.head
    }

    fn refresh_head(&mut self) {
        use std::os::unix::fs::FileExt as _;
        let want = self.offset.min(HEAD_BYTES) as usize;
        if want <= self.head.len() {
            return;
        }
        if let Some(file) = self.file.as_ref() {
            let mut buf = vec![0u8; want];
            if matches!(file.read_at(&mut buf, 0), Ok(n) if n == want) {
                self.head = buf;
            }
        }
    }

    /// The same file was truncated (and maybe rewritten): read it afresh.
    fn restart_in_place(&mut self) {
        if let Some(f) = self.file.as_mut() {
            if f.seek(SeekFrom::Start(0)).is_ok() {
                self.offset = 0;
                self.partial.clear();
                self.head.clear();
            }
        }
    }

    fn drain(&mut self, out: &mut Vec<SpoolRecord>) {
        let Some(file) = self.file.as_mut() else {
            return;
        };
        let mut chunk = [0u8; 8192];
        loop {
            match file.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    self.offset += n as u64;
                    self.partial.extend_from_slice(&chunk[..n]);
                }
            }
        }
        while let Some(pos) = self.partial.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.partial.drain(..=pos).collect();
            if let Ok(text) = std::str::from_utf8(&line) {
                if let Ok(rec) = decode_line::<SpoolRecord>(text) {
                    out.push(rec);
                }
            }
        }
        if self.partial.len() > MAX_LINE_BYTES {
            self.partial.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voicelink_proto::{AgentEvent, AgentKind, EventKind};

    fn rec(text: &str) -> SpoolRecord {
        rec_at(1, text)
    }

    fn rec_at(ts_ms: u64, text: &str) -> SpoolRecord {
        SpoolRecord::new(AgentEvent {
            ts_ms,
            pane: "%1".into(),
            window: Some(0),
            agent: AgentKind::Claude,
            kind: EventKind::TurnEnd,
            text: Some(text.into()),
        })
    }

    fn texts(v: Vec<SpoolRecord>) -> Vec<String> {
        v.into_iter().filter_map(|r| r.event.text).collect()
    }

    #[test]
    fn starts_at_the_end_and_follows_growth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state/events.jsonl");
        append(&path, &rec("old")).unwrap();
        let mut tail = SpoolTail::new(path.clone());
        assert!(tail.poll().is_empty(), "history must not be replayed");
        append(&path, &rec("a")).unwrap();
        append(&path, &rec("b")).unwrap();
        assert_eq!(texts(tail.poll()), ["a", "b"]);
        assert!(tail.poll().is_empty());
    }

    #[test]
    fn file_created_after_start_is_read_from_its_beginning() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let mut tail = SpoolTail::new(path.clone());
        assert!(tail.poll().is_empty());
        append(&path, &rec("first")).unwrap();
        assert_eq!(texts(tail.poll()), ["first"]);
    }

    #[test]
    fn survives_rename_rotation_without_losing_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        append(&path, &rec("seed")).unwrap();
        let mut tail = SpoolTail::new(path.clone());
        append(&path, &rec("before")).unwrap();
        // Rotation happens before the tail polls: the line written to the old
        // inode must still be delivered, then the new file.
        fs::rename(&path, dir.path().join("events.jsonl.1")).unwrap();
        append(&path, &rec("after")).unwrap();
        assert_eq!(texts(tail.poll()), ["before", "after"]);
        append(&path, &rec("later")).unwrap();
        assert_eq!(texts(tail.poll()), ["later"]);
    }

    #[test]
    fn survives_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        append(&path, &rec("a-long-first-record-to-push-the-offset")).unwrap();
        let mut tail = SpoolTail::new(path.clone());
        append(&path, &rec("x")).unwrap();
        assert_eq!(texts(tail.poll()), ["x"]);
        fs::write(&path, "").unwrap();
        assert!(tail.poll().is_empty());
        append(&path, &rec("y")).unwrap();
        assert_eq!(texts(tail.poll()), ["y"]);
    }

    #[test]
    fn detects_truncate_and_rewrite_to_a_larger_size() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        append(&path, &rec_at(1_793_102_500_000, "seed")).unwrap();
        let mut tail = SpoolTail::new(path.clone());
        append(&path, &rec_at(1_793_102_500_001, "short")).unwrap();
        assert_eq!(texts(tail.poll()), ["short"]);
        // Truncate and write more than was there before the next poll.
        fs::write(&path, "").unwrap();
        append(
            &path,
            &rec_at(
                1_793_102_600_000,
                "a considerably longer replacement record than before",
            ),
        )
        .unwrap();
        append(&path, &rec_at(1_793_102_600_001, "and another")).unwrap();
        assert_eq!(
            texts(tail.poll()),
            [
                "a considerably longer replacement record than before",
                "and another"
            ]
        );
    }

    #[test]
    fn skips_malformed_and_waits_for_partial_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        append(&path, &rec("seed")).unwrap();
        let mut tail = SpoolTail::new(path.clone());
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"not json\n{\"v\":1}\n").unwrap();
        let good = encode_line(&rec("ok")).unwrap();
        let (head, rest) = good.split_at(10);
        f.write_all(head.as_bytes()).unwrap();
        assert!(tail.poll().is_empty());
        f.write_all(rest.as_bytes()).unwrap();
        assert_eq!(texts(tail.poll()), ["ok"]);
    }

    #[test]
    fn append_rotates_over_the_limit_and_sets_modes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/events.jsonl");
        for i in 0..5 {
            append_with_limit(&path, &rec(&format!("r{i}")), 200).unwrap();
        }
        let rotated = fs::read_to_string(sibling(&path, ".1")).unwrap();
        let current = fs::read_to_string(&path).unwrap();
        assert!(!rotated.is_empty() && !current.is_empty());
        assert!(rotated.lines().count() + current.lines().count() <= 5);
        for line in rotated.lines().chain(current.lines()) {
            decode_line::<SpoolRecord>(line).unwrap();
        }
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::metadata(path.parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
    }

    #[test]
    fn concurrent_appenders_do_not_interleave() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let p = path.clone();
                std::thread::spawn(move || {
                    for i in 0..25 {
                        append_with_limit(&p, &rec(&format!("t{t}-{i}-{}", "x".repeat(300))), 4000)
                            .unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let mut n = 0;
        for name in ["events.jsonl", "events.jsonl.1"] {
            if let Ok(s) = fs::read_to_string(dir.path().join(name)) {
                for line in s.lines() {
                    decode_line::<SpoolRecord>(line).expect("torn line");
                    n += 1;
                }
            }
        }
        assert!(n > 0);
    }
}
