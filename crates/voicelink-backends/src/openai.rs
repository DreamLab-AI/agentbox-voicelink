//! OpenAI-compatible HTTP backends: transcriptions, chat completions, speech.

use crate::audio::{decode_wav, encode_wav};
use crate::{
    AudioClip, BackendError, ChatMessage, ChatModel, ChatRequest, Result, Role, SpeechToText,
    TextToSpeech, ToolCall, Transcript, resample,
};
use async_trait::async_trait;
use futures_util::{
    StreamExt,
    future::{Either, select},
};
use serde_json::{Map, Value, json};
use std::time::Duration;
use tokio::sync::mpsc;

/// Sample rate assumed when a speech server omits `X-Sample-Rate`.
const DEFAULT_TTS_RATE: u32 = 24_000;
/// Audio delivered to the sink per chunk.
const CHUNK_MS: u32 = 100;

fn http_client(timeout: Option<Duration>) -> Result<reqwest::Client> {
    let mut b = reqwest::Client::builder().connect_timeout(Duration::from_secs(10));
    if let Some(t) = timeout {
        b = b.timeout(t);
    }
    b.build()
        .map_err(|e| BackendError::Config(format!("http client: {e}")))
}

/// Validates the base URL and strips any trailing slash.
fn base(url: &str) -> Result<String> {
    let u = reqwest::Url::parse(url)
        .map_err(|e| BackendError::Config(format!("bad base_url {url:?}: {e}")))?;
    if !matches!(u.scheme(), "http" | "https") {
        return Err(BackendError::Config(format!(
            "base_url {url:?} must be http or https"
        )));
    }
    Ok(url.trim_end_matches('/').to_owned())
}

fn bearer(env: &Option<String>) -> Result<Option<String>> {
    match env {
        None => Ok(None),
        Some(name) => match std::env::var(name) {
            Ok(v) if !v.trim().is_empty() => Ok(Some(v.trim().to_owned())),
            _ => Err(BackendError::Config(format!(
                "api_key_env {name} is unset or empty"
            ))),
        },
    }
}

fn transport(e: reqwest::Error) -> BackendError {
    BackendError::Transport(e.to_string())
}

async fn check(resp: reqwest::Response) -> Result<reqwest::Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp.text().await.unwrap_or_default();
    let body: String = body.chars().take(300).collect();
    Err(BackendError::Backend(format!("HTTP {status}: {body}")))
}

fn authed(rb: reqwest::RequestBuilder, key: &Option<String>) -> reqwest::RequestBuilder {
    match key {
        Some(k) => rb.bearer_auth(k),
        None => rb,
    }
}

// ---------------------------------------------------------------- STT

pub(crate) struct OpenAiStt {
    client: reqwest::Client,
    url: String,
    model: String,
    key: Option<String>,
}

impl OpenAiStt {
    pub(crate) fn new(base_url: &str, model: &str, api_key_env: &Option<String>) -> Result<Self> {
        Ok(Self {
            client: http_client(Some(Duration::from_secs(120)))?,
            url: format!("{}/audio/transcriptions", base(base_url)?),
            model: model.to_owned(),
            key: bearer(api_key_env)?,
        })
    }
}

#[async_trait]
impl SpeechToText for OpenAiStt {
    async fn transcribe(&self, clip: &AudioClip) -> Result<Transcript> {
        let wav = encode_wav(&resample(clip, 16_000));
        let file = reqwest::multipart::Part::bytes(wav)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .map_err(|e| BackendError::Backend(e.to_string()))?;
        let form = reqwest::multipart::Form::new()
            .part("file", file)
            .text("model", self.model.clone())
            .text("response_format", "json");
        let resp = authed(self.client.post(&self.url), &self.key)
            .multipart(form)
            .send()
            .await
            .map_err(transport)?;
        let v: Value =
            check(resp).await?.json().await.map_err(|e| {
                BackendError::Backend(format!("unparseable transcription reply: {e}"))
            })?;
        let text = v
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| BackendError::Backend("transcription reply has no `text`".into()))?;
        Ok(Transcript {
            text: text.trim().to_owned(),
        })
    }

    fn name(&self) -> String {
        format!("openai-stt:{}", self.model)
    }
}

// ---------------------------------------------------------------- chat

pub(crate) struct OpenAiChat {
    client: reqwest::Client,
    url: String,
    model: String,
    key: Option<String>,
    extra: Map<String, Value>,
}

impl OpenAiChat {
    pub(crate) fn new(
        base_url: &str,
        model: &str,
        api_key_env: &Option<String>,
        extra: &Map<String, Value>,
    ) -> Result<Self> {
        Ok(Self {
            client: http_client(Some(Duration::from_secs(300)))?,
            url: format!("{}/chat/completions", base(base_url)?),
            model: model.to_owned(),
            key: bearer(api_key_env)?,
            extra: extra.clone(),
        })
    }
}

fn role_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// OpenAI wants arguments as a JSON string; a raw string that failed to parse on
/// the way in goes back out unchanged rather than double-quoted.
fn arguments_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn message_json(m: &ChatMessage) -> Value {
    let mut o = Map::new();
    o.insert("role".into(), json!(role_str(m.role)));
    if m.content.is_some() || m.tool_calls.is_empty() {
        o.insert(
            "content".into(),
            json!(m.content.clone().unwrap_or_default()),
        );
    } else {
        o.insert("content".into(), Value::Null);
    }
    if !m.tool_calls.is_empty() {
        let calls: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|c| {
                json!({
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": arguments_string(&c.arguments)},
                })
            })
            .collect();
        o.insert("tool_calls".into(), Value::Array(calls));
    }
    if let Some(id) = &m.tool_call_id {
        o.insert("tool_call_id".into(), json!(id));
    }
    Value::Object(o)
}

pub(crate) fn request_body(model: &str, extra: &Map<String, Value>, req: &ChatRequest) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert(
        "messages".into(),
        Value::Array(req.messages.iter().map(message_json).collect()),
    );
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| {
                json!({"type": "function", "function": {
                    "name": t.name, "description": t.description, "parameters": t.parameters,
                }})
            })
            .collect();
        body.insert("tools".into(), Value::Array(tools));
    }
    body.insert("temperature".into(), json!(req.temperature));
    body.insert("max_tokens".into(), json!(req.max_tokens));
    for (k, v) in extra {
        body.insert(k.clone(), v.clone());
    }
    Value::Object(body)
}

fn parse_arguments(v: Option<&Value>) -> Value {
    match v {
        None | Some(Value::Null) => json!({}),
        Some(Value::String(s)) if s.trim().is_empty() => json!({}),
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(other) => other.clone(),
    }
}

fn content_text(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let t: String = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect();
            (!t.is_empty()).then_some(t)
        }
        _ => None,
    }
}

pub(crate) fn parse_chat_reply(v: &Value) -> Result<ChatMessage> {
    let msg = v
        .pointer("/choices/0/message")
        .ok_or_else(|| BackendError::Backend("chat reply has no choices[0].message".into()))?;
    let mut tool_calls = Vec::new();
    if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
        for (i, c) in calls.iter().enumerate() {
            let f = c.get("function").unwrap_or(c);
            let name = f
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| BackendError::Backend("tool call without a name".into()))?;
            let id = c
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map_or_else(|| format!("call_{i}"), str::to_owned);
            tool_calls.push(ToolCall {
                id,
                name: name.to_owned(),
                arguments: parse_arguments(f.get("arguments")),
            });
        }
    }
    // Servers often send "" beside tool calls; the contract says absent.
    let content = content_text(msg.get("content"))
        .filter(|c| !(!tool_calls.is_empty() && c.trim().is_empty()));
    Ok(ChatMessage {
        role: Role::Assistant,
        content,
        tool_calls,
        tool_call_id: None,
    })
}

#[async_trait]
impl ChatModel for OpenAiChat {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatMessage> {
        let body = request_body(&self.model, &self.extra, request);
        let resp = authed(self.client.post(&self.url), &self.key)
            .json(&body)
            .send()
            .await
            .map_err(transport)?;
        let v: Value = check(resp)
            .await?
            .json()
            .await
            .map_err(|e| BackendError::Backend(format!("unparseable chat reply: {e}")))?;
        parse_chat_reply(&v)
    }

    fn name(&self) -> String {
        format!("openai-chat:{}", self.model)
    }
}

// ---------------------------------------------------------------- TTS

pub(crate) struct OpenAiTts {
    client: reqwest::Client,
    url: String,
    model: String,
    voice: String,
    key: Option<String>,
}

impl OpenAiTts {
    pub(crate) fn new(
        base_url: &str,
        model: &str,
        voice: &str,
        api_key_env: &Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            client: http_client(None)?,
            url: format!("{}/audio/speech", base(base_url)?),
            model: model.to_owned(),
            voice: voice.to_owned(),
            key: bearer(api_key_env)?,
        })
    }
}

/// Sends `samples` to the sink in ~100 ms chunks.
async fn send_clip(sink: &mpsc::Sender<AudioClip>, rate: u32, samples: Vec<f32>) -> Result<()> {
    let step = (rate * CHUNK_MS / 1000).max(1) as usize;
    for c in samples.chunks(step) {
        sink.send(AudioClip {
            sample_rate: rate,
            samples: c.to_vec(),
        })
        .await
        .map_err(|_| BackendError::Cancelled)?;
    }
    Ok(())
}

#[async_trait]
impl TextToSpeech for OpenAiTts {
    async fn synthesize(&self, text: &str, sink: mpsc::Sender<AudioClip>) -> Result<()> {
        if sink.is_closed() {
            return Err(BackendError::Cancelled);
        }
        let body = json!({
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "stream": true,
        });
        let send = authed(self.client.post(&self.url), &self.key)
            .json(&body)
            .send();
        let resp = match select(Box::pin(send), Box::pin(sink.closed())).await {
            Either::Left((r, _)) => r.map_err(transport)?,
            Either::Right(_) => return Err(BackendError::Cancelled),
        };
        let resp = check(resp).await?;
        let ctype = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        let rate = resp
            .headers()
            .get("x-sample-rate")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|r| *r > 0)
            .unwrap_or(DEFAULT_TTS_RATE);
        let mut stream = resp.bytes_stream();

        if ctype.contains("wav") {
            // The server ignored `pcm`/`stream`: take the complete file.
            let mut all = Vec::new();
            loop {
                match select(Box::pin(stream.next()), Box::pin(sink.closed())).await {
                    Either::Left((Some(b), _)) => all.extend_from_slice(&b.map_err(transport)?),
                    Either::Left((None, _)) => break,
                    Either::Right(_) => return Err(BackendError::Cancelled),
                }
            }
            let clip = decode_wav(&all)?;
            return send_clip(&sink, clip.sample_rate, clip.samples).await;
        }
        if !(ctype.is_empty()
            || ctype.contains("pcm")
            || ctype.contains("octet-stream")
            || ctype.contains("l16"))
        {
            return Err(BackendError::Backend(format!(
                "unsupported speech content type {ctype:?}"
            )));
        }

        let step = (rate * CHUNK_MS / 1000).max(1) as usize;
        let mut carry: Option<u8> = None;
        let mut pending: Vec<f32> = Vec::with_capacity(step * 2);
        loop {
            let next = match select(Box::pin(stream.next()), Box::pin(sink.closed())).await {
                Either::Left((n, _)) => n,
                Either::Right(_) => return Err(BackendError::Cancelled),
            };
            let Some(bytes) = next else { break };
            let bytes = bytes.map_err(transport)?;
            let mut it = bytes.iter().copied();
            if let Some(lo) = carry.take() {
                match it.next() {
                    Some(hi) => pending.push(i16::from_le_bytes([lo, hi]) as f32 / 32768.0),
                    None => carry = Some(lo),
                }
            }
            let rest: Vec<u8> = it.collect();
            let mut pairs = rest.chunks_exact(2);
            for p in &mut pairs {
                pending.push(i16::from_le_bytes([p[0], p[1]]) as f32 / 32768.0);
            }
            if let [b] = pairs.remainder() {
                carry = Some(*b);
            }
            while pending.len() >= step {
                let out: Vec<f32> = pending.drain(..step).collect();
                send_clip(&sink, rate, out).await?;
            }
        }
        if !pending.is_empty() {
            send_clip(&sink, rate, pending).await?;
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("openai-tts:{}:{}", self.model, self.voice)
    }
}
