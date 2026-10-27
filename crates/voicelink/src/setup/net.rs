//! HTTP helpers: local LLM server detection, the tool-calling probe, and
//! resumable, hash-verified model downloads.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, bail};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::catalogue::Entry;

fn agent(connect: Duration, global: Option<Duration>) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(connect))
        .timeout_global(global)
        .http_status_as_error(false)
        .build()
        .into()
}

// ---------------------------------------------------------------- detection

/// An OpenAI-compatible server found on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectedServer {
    /// Product name.
    pub label: String,
    /// Base URL up to and including `/v1`.
    pub base_url: String,
    /// Model ids from `GET /v1/models`.
    pub models: Vec<String>,
}

/// Default ports of the servers we look for.
pub const CANDIDATES: [(&str, &str); 3] = [
    ("llama-server", "http://127.0.0.1:8080/v1"),
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
];

/// Model ids from a `/v1/models` reply.
pub fn parse_models(v: &Value) -> Vec<String> {
    v.get("data")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|m| m.get("id").and_then(Value::as_str).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Probes the well-known localhost ports in parallel.
pub fn detect_llm_servers() -> Vec<DetectedServer> {
    let handles: Vec<_> = CANDIDATES
        .iter()
        .map(|(label, base)| {
            let (label, base) = (label.to_string(), base.to_string());
            std::thread::spawn(move || {
                let a = agent(Duration::from_millis(600), Some(Duration::from_secs(3)));
                let mut resp = a.get(&format!("{base}/models")).call().ok()?;
                if !resp.status().is_success() {
                    return None;
                }
                let v: Value = resp.body_mut().read_json().ok()?;
                Some(DetectedServer {
                    label,
                    base_url: base,
                    models: parse_models(&v),
                })
            })
        })
        .collect();
    handles
        .into_iter()
        .filter_map(|h| h.join().ok().flatten())
        .collect()
}

// --------------------------------------------------------------- tool probe

/// Outcome of the tool-calling probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolProbe {
    /// The server returned a structured `tool_calls` entry.
    Structured {
        /// Name of the tool the model called.
        tool: String,
    },
    /// A normal reply without `tool_calls`.
    NoToolCalls {
        /// Start of the reply text.
        reply: String,
        /// The reply looks like a tool call serialised into the text.
        looks_like_text_call: bool,
    },
    /// Transport or HTTP failure.
    Failed(String),
}

impl ToolProbe {
    /// One-paragraph explanation for the operator.
    pub fn explain(&self) -> String {
        match self {
            ToolProbe::Structured { tool } => format!("OK: the server returned a structured tool call ({tool})."),
            ToolProbe::NoToolCalls { looks_like_text_call: true, .. } => "WARNING: the model wrote a tool call as plain text instead of returning structured tool_calls. \
                 llama-server needs --jinja (and a model whose chat template supports tools); the concierge cannot act without it."
                .into(),
            ToolProbe::NoToolCalls { reply, .. } => format!(
                "WARNING: the server returned no structured tool_calls (reply: {reply:?}). \
                 If this is llama-server, start it with --jinja; otherwise pick a model with tool-calling support. The concierge cannot act without it."
            ),
            ToolProbe::Failed(e) => format!("The probe request failed: {e}"),
        }
    }
}

/// The probe request body: one trivial tool the model is told to call.
pub fn tool_probe_request(model: &str, extra: &Map<String, Value>) -> Value {
    let mut body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "Use the add tool to compute 2 + 3. Do not answer in text."}],
        "tools": [{"type": "function", "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}
        }}],
        "tool_choice": "auto",
        "temperature": 0,
        // Reasoning models spend tokens on thinking before the call.
        "max_tokens": 1536
    });
    if let Some(o) = body.as_object_mut() {
        for (k, v) in extra {
            o.insert(k.clone(), v.clone());
        }
    }
    body
}

/// Classifies a `/chat/completions` reply.
pub fn classify_tool_response(v: &Value) -> ToolProbe {
    let msg = &v["choices"][0]["message"];
    if let Some(call) = msg
        .get("tool_calls")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
    {
        let tool = call["function"]["name"].as_str().unwrap_or("?").to_string();
        return ToolProbe::Structured { tool };
    }
    if msg.is_null() {
        let detail = v
            .get("error")
            .map(Value::to_string)
            .unwrap_or_else(|| "no choices in reply".into());
        return ToolProbe::Failed(detail);
    }
    let text = msg
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let looks = text.contains("<tool_call>")
        || text.contains("\"name\"") && text.contains("add")
        || text.contains("[TOOL_CALLS]");
    ToolProbe::NoToolCalls {
        reply: text.chars().take(120).collect(),
        looks_like_text_call: looks,
    }
}

/// Sends the probe to `base_url`.
pub fn tool_probe(
    base_url: &str,
    model: &str,
    extra: &Map<String, Value>,
    api_key: Option<&str>,
) -> ToolProbe {
    let a = agent(Duration::from_secs(5), Some(Duration::from_secs(120)));
    let mut req = a.post(&format!(
        "{}/chat/completions",
        base_url.trim_end_matches('/')
    ));
    if let Some(k) = api_key {
        req = req.header("Authorization", &format!("Bearer {k}"));
    }
    match req.send_json(tool_probe_request(model, extra)) {
        Ok(mut resp) => {
            let status = resp.status();
            match resp.body_mut().read_json::<Value>() {
                Ok(v) if status.is_success() => classify_tool_response(&v),
                Ok(v) => ToolProbe::Failed(format!("HTTP {status}: {v}")),
                Err(e) => ToolProbe::Failed(format!("HTTP {status}, unreadable body: {e}")),
            }
        }
        Err(e) => ToolProbe::Failed(e.to_string()),
    }
}

// ----------------------------------------------------------------- download

/// Lower-case hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// SHA-256 of a file, hex.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 18];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Downloads a catalogue entry into `dir`.
pub fn download(
    entry: &Entry,
    dir: &Path,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, u64),
) -> anyhow::Result<PathBuf> {
    download_url(entry.url, entry.sha256, entry.file, dir, cancel, progress)
}

/// Downloads `url` to `dir/file`: resumes `file.part` with an HTTP Range
/// request, verifies `sha256`, then renames atomically. An existing complete
/// file is verified and kept.
pub fn download_url(
    url: &str,
    sha256: &str,
    file: &str,
    dir: &Path,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(u64, u64),
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let dest = dir.join(file);
    let part = dir.join(format!("{file}.part"));
    if dest.exists() {
        if sha256_file(&dest)? == sha256 {
            let n = std::fs::metadata(&dest)?.len();
            progress(n, n);
            return Ok(dest);
        }
        std::fs::remove_file(&dest)?;
    }
    for attempt in 0..2 {
        let offset = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let a = agent(Duration::from_secs(15), None);
        let mut req = a.get(url);
        if offset > 0 {
            req = req.header("Range", &format!("bytes={offset}-"));
        }
        let resp = req.call().with_context(|| format!("requesting {url}"))?;
        let status = resp.status().as_u16();
        let (mut out, start) = match status {
            206 => (
                std::fs::OpenOptions::new().append(true).open(&part)?,
                offset,
            ),
            200 => (std::fs::File::create(&part)?, 0),
            416 => {
                // Our partial file is already complete (or longer than the source): verify it.
                if sha256_file(&part)? == sha256 {
                    std::fs::rename(&part, &dest)?;
                    return Ok(dest);
                }
                std::fs::remove_file(&part)?;
                if attempt == 0 {
                    continue;
                }
                bail!("server rejected the resume range twice for {url}");
            }
            s => bail!("HTTP {s} fetching {url}"),
        };
        let remaining = resp
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        let total = start + remaining;
        let mut body = resp.into_body().into_reader();
        let mut buf = vec![0u8; 1 << 17];
        let mut done = start;
        progress(done, total);
        loop {
            if cancel.load(Ordering::Relaxed) {
                out.flush()?;
                bail!("cancelled (partial file kept for resume)");
            }
            let n = body
                .read(&mut buf)
                .with_context(|| format!("reading {url}"))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            done += n as u64;
            progress(done, total.max(done));
        }
        out.sync_all()?;
        drop(out);
        let got = sha256_file(&part)?;
        if got != sha256 {
            let _ = std::fs::remove_file(&part);
            bail!(
                "SHA-256 mismatch for {file}: expected {sha256}, got {got}; partial file removed"
            );
        }
        std::fs::rename(&part, &dest)?;
        return Ok(dest);
    }
    bail!("download of {url} failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    #[test]
    fn models_parse() {
        let v = json!({"object": "list", "data": [{"id": "qwen"}, {"id": "llama"}]});
        assert_eq!(parse_models(&v), vec!["qwen", "llama"]);
        assert!(parse_models(&json!({})).is_empty());
    }

    #[test]
    fn tool_probe_classification() {
        let ok = json!({"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "add", "arguments": "{\"a\":2,\"b\":3}"}}]}}]});
        assert_eq!(
            classify_tool_response(&ok),
            ToolProbe::Structured { tool: "add".into() }
        );
        let text = json!({"choices": [{"message": {"content": "<tool_call>{\"name\": \"add\"}</tool_call>"}}]});
        let r = classify_tool_response(&text);
        assert!(matches!(
            r,
            ToolProbe::NoToolCalls {
                looks_like_text_call: true,
                ..
            }
        ));
        assert!(r.explain().contains("--jinja"));
        let plain = json!({"choices": [{"message": {"content": "5"}}]});
        assert!(matches!(
            classify_tool_response(&plain),
            ToolProbe::NoToolCalls {
                looks_like_text_call: false,
                ..
            }
        ));
        assert!(classify_tool_response(&plain).explain().contains("--jinja"));
        assert!(matches!(
            classify_tool_response(&json!({"error": {"message": "x"}})),
            ToolProbe::Failed(_)
        ));
    }

    #[test]
    fn probe_request_merges_extra() {
        let extra = crate::setup::choices::loom_extra();
        let body = tool_probe_request("m", &extra);
        assert_eq!(body["loom_options"]["scaffold"], json!(false));
        assert_eq!(body["tools"][0]["function"]["name"], "add");
        assert!(body["max_tokens"].as_u64().unwrap() >= 1536);
    }

    /// Serves `data`, honouring `Range: bytes=N-`, for `requests` connections.
    fn serve(
        data: Vec<u8>,
        requests: usize,
        honour_range: bool,
    ) -> (String, std::thread::JoinHandle<Vec<Option<u64>>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/f.bin", l.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for _ in 0..requests {
                let (mut s, _) = l.accept().unwrap();
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut range = None;
                loop {
                    let mut line = String::new();
                    r.read_line(&mut line).unwrap();
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                        range = v.trim().trim_end_matches('-').parse::<u64>().ok();
                    }
                }
                seen.push(range);
                match range.filter(|_| honour_range) {
                    Some(o) if o as usize >= data.len() => {
                        write!(s, "HTTP/1.1 416 Range Not Satisfiable\r\ncontent-length: 0\r\nconnection: close\r\n\r\n").unwrap();
                    }
                    Some(o) => {
                        let body = &data[o as usize..];
                        write!(s, "HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\ncontent-range: bytes {o}-{}/{}\r\nconnection: close\r\n\r\n", body.len(), data.len() - 1, data.len()).unwrap();
                        s.write_all(body).unwrap();
                    }
                    None => {
                        write!(
                            s,
                            "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                            data.len()
                        )
                        .unwrap();
                        s.write_all(&data).unwrap();
                    }
                }
            }
            seen
        });
        (url, h)
    }

    fn sample() -> (Vec<u8>, String) {
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
        let sum = hex(&Sha256::digest(&data));
        (data, sum)
    }

    #[test]
    fn downloads_and_verifies() {
        let (data, sum) = sample();
        let (url, h) = serve(data.clone(), 1, true);
        let dir = tempfile::tempdir().unwrap();
        let mut last = (0, 0);
        let p = download_url(
            &url,
            &sum,
            "f.bin",
            dir.path(),
            &AtomicBool::new(false),
            &mut |d, t| last = (d, t),
        )
        .unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), data);
        assert_eq!(last, (data.len() as u64, data.len() as u64));
        assert!(!dir.path().join("f.bin.part").exists());
        h.join().unwrap();
        // A second call is satisfied from disk without any request (no server needed).
        download_url(
            &url,
            &sum,
            "f.bin",
            dir.path(),
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap();
    }

    #[test]
    fn resumes_with_range() {
        let (data, sum) = sample();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.bin.part"), &data[..100_000]).unwrap();
        let (url, h) = serve(data.clone(), 1, true);
        download_url(
            &url,
            &sum,
            "f.bin",
            dir.path(),
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(h.join().unwrap(), vec![Some(100_000)]);
        assert_eq!(std::fs::read(dir.path().join("f.bin")).unwrap(), data);
    }

    #[test]
    fn restarts_when_server_ignores_range() {
        let (data, sum) = sample();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.bin.part"), &data[..1000]).unwrap();
        let (url, h) = serve(data.clone(), 1, false);
        download_url(
            &url,
            &sum,
            "f.bin",
            dir.path(),
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap();
        h.join().unwrap();
        assert_eq!(std::fs::read(dir.path().join("f.bin")).unwrap(), data);
    }

    #[test]
    fn complete_partial_file_is_accepted_on_416() {
        let (data, sum) = sample();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.bin.part"), &data).unwrap();
        let (url, h) = serve(data.clone(), 1, true);
        download_url(
            &url,
            &sum,
            "f.bin",
            dir.path(),
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap();
        h.join().unwrap();
        assert!(dir.path().join("f.bin").exists());
    }

    #[test]
    fn hash_mismatch_is_rejected_and_cleaned() {
        let (data, _) = sample();
        let (url, h) = serve(data, 1, true);
        let dir = tempfile::tempdir().unwrap();
        let err = download_url(
            &url,
            &"0".repeat(64),
            "f.bin",
            dir.path(),
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("mismatch"), "{err}");
        h.join().unwrap();
        assert!(!dir.path().join("f.bin").exists());
        assert!(!dir.path().join("f.bin.part").exists());
    }

    #[test]
    fn corrupt_existing_file_is_replaced() {
        let (data, sum) = sample();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.bin"), b"junk").unwrap();
        let (url, h) = serve(data.clone(), 1, true);
        download_url(
            &url,
            &sum,
            "f.bin",
            dir.path(),
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap();
        h.join().unwrap();
        assert_eq!(std::fs::read(dir.path().join("f.bin")).unwrap(), data);
    }
}
