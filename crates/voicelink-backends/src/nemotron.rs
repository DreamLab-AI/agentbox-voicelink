//! Nemotron streaming ASR over WebSocket + MessagePack.
//!
//! Protocol (`/api/asr-streaming`): the server sends `Ready`; the client sends
//! `{"type":"Audio","pcm":[f32…]}` frames of 80 ms at 24 kHz, then
//! `{"type":"Marker","id":n}` followed by at least 0.5 s of silence; the server
//! streams `Word{text}` messages and echoes the `Marker` once every word of the
//! utterance has been emitted.

use crate::{AudioClip, BackendError, Result, SpeechToText, Transcript, resample};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt, future::try_join};
use serde::Serialize;
use serde_json::Value;
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

/// Sample rate the server expects.
const INPUT_RATE: u32 = 24_000;
/// Samples per 80 ms frame.
const FRAME: usize = (INPUT_RATE as usize * 80) / 1000;
/// Trailing silence frames (7 x 80 ms = 0.56 s) that flush the final tokens.
const FLUSH_FRAMES: usize = 7;
/// Longest utterance the Nemotron backend accepts. The server's audio queue
/// holds about 36 s and overflows rather than blocks, so longer clips would
/// lose audio silently; callers segment speech below this limit.
pub const NEMOTRON_MAX_UTTERANCE: Duration = Duration::from_secs(30);
/// Longest silence tolerated while awaiting the marker echo.
const REPLY_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Serialize)]
struct AudioMsg<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    pcm: &'a [f32],
}

#[derive(Serialize)]
struct MarkerMsg {
    #[serde(rename = "type")]
    kind: &'static str,
    id: u32,
}

pub(crate) struct NemotronStt {
    url: String,
}

impl NemotronStt {
    pub(crate) fn new(url: &str) -> Result<Self> {
        let u = reqwest::Url::parse(url)
            .map_err(|e| BackendError::Config(format!("bad Nemotron url {url:?}: {e}")))?;
        if !matches!(u.scheme(), "ws" | "wss") {
            return Err(BackendError::Config(format!(
                "Nemotron url {url:?} must be ws:// or wss://"
            )));
        }
        Ok(Self {
            url: url.to_owned(),
        })
    }
}

fn pack<T: Serialize>(m: &T) -> Result<Message> {
    rmp_serde::to_vec_named(m)
        .map(|v| Message::Binary(v.into()))
        .map_err(|e| BackendError::Backend(format!("msgpack encode: {e}")))
}

fn unpack(m: &Message) -> Result<Option<Value>> {
    match m {
        Message::Binary(b) => rmp_serde::from_slice::<Value>(b)
            .map(Some)
            .map_err(|e| BackendError::Backend(format!("msgpack decode: {e}"))),
        Message::Close(_) => Err(BackendError::Transport(
            "server closed the connection".into(),
        )),
        _ => Ok(None),
    }
}

fn ws_err(e: tokio_tungstenite::tungstenite::Error) -> BackendError {
    BackendError::Transport(e.to_string())
}

/// Server error text from an `Error` message, if `v` is one.
fn server_error(v: &Value) -> Option<BackendError> {
    (v.get("type").and_then(Value::as_str) == Some("Error")).then(|| {
        BackendError::Backend(format!(
            "nemotron: {}",
            v.get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        ))
    })
}

#[async_trait]
impl SpeechToText for NemotronStt {
    async fn transcribe(&self, clip: &AudioClip) -> Result<Transcript> {
        let seconds = clip.samples.len() as f64 / f64::from(clip.sample_rate.max(1));
        if seconds > NEMOTRON_MAX_UTTERANCE.as_secs_f64() {
            return Err(BackendError::Backend(format!(
                "utterance of {seconds:.1} s exceeds the Nemotron limit of {} s",
                NEMOTRON_MAX_UTTERANCE.as_secs()
            )));
        }
        let mut samples = resample(clip, INPUT_RATE).samples;
        samples.resize(samples.len().div_ceil(FRAME) * FRAME, 0.0);
        samples.resize(samples.len() + FLUSH_FRAMES * FRAME, 0.0);
        let speech_frames = samples.len() / FRAME - FLUSH_FRAMES;

        let (ws, _) = tokio::time::timeout(
            Duration::from_secs(10),
            tokio_tungstenite::connect_async(self.url.as_str()),
        )
        .await
        .map_err(|_| BackendError::Transport("connect timed out".into()))?
        .map_err(ws_err)?;
        let (mut tx, mut rx) = ws.split();

        // Wait for Ready.
        loop {
            let m = tokio::time::timeout(REPLY_TIMEOUT, rx.next())
                .await
                .map_err(|_| BackendError::Transport("timed out waiting for Ready".into()))?
                .ok_or_else(|| BackendError::Transport("closed before Ready".into()))?
                .map_err(ws_err)?;
            if let Some(v) = unpack(&m)? {
                if let Some(e) = server_error(&v) {
                    return Err(e);
                }
                if v.get("type").and_then(Value::as_str) == Some("Ready") {
                    break;
                }
            }
        }

        const MARKER_ID: u32 = 1;
        let send = async {
            for (i, f) in samples.chunks(FRAME).enumerate() {
                if i == speech_frames {
                    tx.send(pack(&MarkerMsg {
                        kind: "Marker",
                        id: MARKER_ID,
                    })?)
                    .await
                    .map_err(ws_err)?;
                }
                tx.send(pack(&AudioMsg {
                    kind: "Audio",
                    pcm: f,
                })?)
                .await
                .map_err(ws_err)?;
                // The server's queue holds roughly 36 s of audio and overflows
                // rather than blocks: send ~8x real time for the first 24 s,
                // then drop to 2x so long utterances cannot outrun inference.
                let gap = if i < 300 { 10 } else { 40 };
                tokio::time::sleep(Duration::from_millis(gap)).await;
            }
            Ok::<_, BackendError>(())
        };
        let recv = async {
            let mut text = String::new();
            loop {
                let m = tokio::time::timeout(REPLY_TIMEOUT, rx.next())
                    .await
                    .map_err(|_| BackendError::Transport("timed out waiting for Marker".into()))?
                    .ok_or_else(|| BackendError::Transport("closed before Marker".into()))?
                    .map_err(ws_err)?;
                let Some(v) = unpack(&m)? else { continue };
                if let Some(e) = server_error(&v) {
                    return Err(e);
                }
                match v.get("type").and_then(Value::as_str) {
                    Some("Word") => {
                        if let Some(t) = v.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                        }
                    }
                    Some("Marker")
                        if v.get("id").and_then(Value::as_u64) == Some(MARKER_ID as u64) =>
                    {
                        return Ok(text);
                    }
                    _ => {}
                }
            }
        };
        let result = try_join(send, recv).await.map(|(_, t)| t);
        let _ = tx.close().await;
        let text = result?;
        Ok(Transcript {
            text: text.split_whitespace().collect::<Vec<_>>().join(" "),
        })
    }

    fn name(&self) -> String {
        "nemotron-stt".into()
    }
}
