//! Interchangeable speech-to-text, chat-model and text-to-speech backends,
//! all reached over the network.
//!
//! Each capability is a trait ([`SpeechToText`], [`ChatModel`],
//! [`TextToSpeech`]) with implementations chosen at runtime from a serde
//! config enum ([`SttConfig`], [`LlmConfig`], [`TtsConfig`]):
//!
//! | Slot | Backends |
//! |---|---|
//! | STT | OpenAI `/v1/audio/transcriptions` (whisper.cpp `whisper-server`, speaches, OpenAI); Nemotron streaming WebSocket |
//! | LLM | OpenAI `/v1/chat/completions` with tool calls (llama-server, Ollama, LM Studio, gateways) |
//! | TTS | OpenAI `/v1/audio/speech` (Kokoro-FastAPI, Pocket TTS) |
//!
//! There are no embedded inference engines: a model server outlives the
//! client, and GPU runtimes (ROCm, CUDA, Vulkan, Metal) belong to that server
//! rather than to a client binary.
//!
//! # Health checks
//!
//! [`probe_stt`], [`probe_llm`] and [`probe_tts`] make one cheap, bounded
//! (about three seconds) reachability check and name the URL in every error,
//! so a launcher can say exactly which endpoint is down before it starts.
//!
//! # Example
//!
//! ```
//! use voicelink_backends::{build_llm, build_stt, LlmConfig, SttConfig};
//!
//! // Servers are contacted lazily, so building never touches the network.
//! let llm = build_llm(&LlmConfig::OpenAi {
//!     base_url: "http://127.0.0.1:8080/v1".into(),
//!     model: "local".into(),
//!     api_key_env: None,
//!     extra: serde_json::Map::new(),
//! })
//! .unwrap();
//! assert_eq!(llm.name(), "openai-chat:local");
//!
//! // A malformed endpoint is a configuration error, not a network one.
//! assert!(build_stt(&SttConfig::Nemotron { url: "http://not-a-websocket".into() }).is_err());
//! ```

mod audio;
mod nemotron;
mod openai;

pub use audio::{decode_wav, encode_wav, resample};
pub use nemotron::NEMOTRON_MAX_UTTERANCE;

/// Longest a [`probe_stt`], [`probe_llm`] or [`probe_tts`] call waits.
pub const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Mono PCM audio, `f32` samples in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioClip {
    /// Samples per second.
    pub sample_rate: u32,
    /// Mono samples.
    pub samples: Vec<f32>,
}

/// A transcription result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// Recognised text, trimmed.
    pub text: String,
}

/// Errors from any backend.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// The backend is misconfigured (bad URL, unset API-key variable).
    #[error("configuration: {0}")]
    Config(String),
    /// Transport failure talking to a server.
    #[error("transport: {0}")]
    Transport(String),
    /// The server answered with an error or an unparseable reply.
    #[error("backend: {0}")]
    Backend(String),
    /// The operation was cancelled (receiver dropped).
    #[error("cancelled")]
    Cancelled,
}

/// Result alias for backend operations.
pub type Result<T> = std::result::Result<T, BackendError>;

/// Speech-to-text.
#[async_trait]
pub trait SpeechToText: Send + Sync {
    /// Transcribes one utterance.
    async fn transcribe(&self, clip: &AudioClip) -> Result<Transcript>;
    /// Short human-readable backend name, for status and logs.
    fn name(&self) -> String;
}

/// Text-to-speech, streamed.
#[async_trait]
pub trait TextToSpeech: Send + Sync {
    /// Synthesises `text`, sending audio chunks to `sink` as they are produced.
    /// Returns [`BackendError::Cancelled`] promptly once `sink` is closed, which
    /// is how a barge-in stops synthesis.
    async fn synthesize(&self, text: &str, sink: mpsc::Sender<AudioClip>) -> Result<()>;
    /// Short human-readable backend name.
    fn name(&self) -> String;
}

/// A chat model with OpenAI-style tool calling.
#[async_trait]
pub trait ChatModel: Send + Sync {
    /// Runs one completion.
    async fn chat(&self, request: &ChatRequest) -> Result<ChatMessage>;
    /// Short human-readable backend name.
    fn name(&self) -> String;
}

/// Chat roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// System prompt.
    System,
    /// The operator.
    User,
    /// The model.
    Assistant,
    /// A tool result.
    Tool,
}

/// One chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Author role.
    pub role: Role,
    /// Text content, absent on a pure tool-call message.
    pub content: Option<String>,
    /// Tool calls requested by the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For [`Role::Tool`]: the call being answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call id, echoed in the tool result.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Parsed JSON arguments.
    pub arguments: serde_json::Value,
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// What the tool does, for the model.
    pub description: String,
    /// JSON Schema of the arguments object.
    pub parameters: serde_json::Value,
}

/// One chat completion request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Conversation so far.
    pub messages: Vec<ChatMessage>,
    /// Tools on offer.
    pub tools: Vec<ToolSpec>,
    /// Sampling temperature.
    pub temperature: f32,
    /// Completion token cap.
    pub max_tokens: u32,
}

/// Speech-to-text backend selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SttConfig {
    /// OpenAI-compatible `/v1/audio/transcriptions`.
    OpenAi {
        /// Base URL up to and including `/v1`.
        base_url: String,
        /// Model name sent in the request.
        model: String,
        /// Environment variable holding a bearer token, if any.
        api_key_env: Option<String>,
    },
    /// Nemotron streaming ASR WebSocket (`/api/asr-streaming`, MessagePack).
    Nemotron {
        /// WebSocket URL.
        url: String,
    },
}

/// Chat model selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LlmConfig {
    /// OpenAI-compatible `/v1/chat/completions` (llama-server, Ollama, LM Studio, the Loom).
    OpenAi {
        /// Base URL up to and including `/v1`.
        base_url: String,
        /// Model name.
        model: String,
        /// Environment variable holding a bearer token, if any.
        api_key_env: Option<String>,
        /// Extra top-level request fields merged verbatim (for example `{"loom_options":{"scaffold":false}}`).
        #[serde(default)]
        extra: serde_json::Map<String, serde_json::Value>,
    },
}

/// Text-to-speech backend selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TtsConfig {
    /// OpenAI-compatible `/v1/audio/speech` (Kokoro-FastAPI, Pocket TTS).
    OpenAi {
        /// Base URL up to and including `/v1`.
        base_url: String,
        /// Model name.
        model: String,
        /// Voice name.
        voice: String,
        /// Environment variable holding a bearer token, if any.
        api_key_env: Option<String>,
    },
}

/// Builds the speech-to-text backend selected by `config`.
///
/// # Errors
///
/// [`BackendError::Config`] when the settings are invalid.
pub fn build_stt(config: &SttConfig) -> Result<Box<dyn SpeechToText>> {
    match config {
        SttConfig::OpenAi {
            base_url,
            model,
            api_key_env,
        } => Ok(Box::new(openai::OpenAiStt::new(
            base_url,
            model,
            api_key_env,
        )?)),
        SttConfig::Nemotron { url } => Ok(Box::new(nemotron::NemotronStt::new(url)?)),
    }
}

/// Builds the chat model selected by `config`.
///
/// # Errors
///
/// [`BackendError::Config`] when the settings are invalid.
pub fn build_llm(config: &LlmConfig) -> Result<Box<dyn ChatModel>> {
    match config {
        LlmConfig::OpenAi {
            base_url,
            model,
            api_key_env,
            extra,
        } => Ok(Box::new(openai::OpenAiChat::new(
            base_url,
            model,
            api_key_env,
            extra,
        )?)),
    }
}

/// Builds the text-to-speech backend selected by `config`.
///
/// # Errors
///
/// [`BackendError::Config`] when the settings are invalid.
pub fn build_tts(config: &TtsConfig) -> Result<Box<dyn TextToSpeech>> {
    match config {
        TtsConfig::OpenAi {
            base_url,
            model,
            voice,
            api_key_env,
        } => Ok(Box::new(openai::OpenAiTts::new(
            base_url,
            model,
            voice,
            api_key_env,
        )?)),
    }
}

/// Checks that the speech-to-text server in `config` is reachable.
///
/// OpenAI-style servers get a `GET {base_url}/models`; Nemotron gets a
/// WebSocket connection that is closed again once the server says `Ready`.
/// Waits at most [`PROBE_TIMEOUT`].
///
/// # Errors
///
/// [`BackendError::Config`] for an invalid setting, [`BackendError::Transport`]
/// when the server cannot be reached and [`BackendError::Backend`] when it
/// answers with an error. Every message names the URL.
pub async fn probe_stt(config: &SttConfig) -> Result<()> {
    match config {
        SttConfig::OpenAi {
            base_url,
            api_key_env,
            ..
        } => openai::probe_models(base_url, api_key_env, true).await,
        SttConfig::Nemotron { url } => nemotron::probe(url).await,
    }
}

/// Checks that the chat server in `config` is reachable (`GET {base_url}/models`).
///
/// Unlike the speech probes, a 404 is an error: every supported chat server
/// lists its models, so a 404 means a wrong base URL (typically a missing
/// `/v1`). Waits at most [`PROBE_TIMEOUT`].
///
/// # Errors
///
/// As for [`probe_stt`].
pub async fn probe_llm(config: &LlmConfig) -> Result<()> {
    match config {
        LlmConfig::OpenAi {
            base_url,
            api_key_env,
            ..
        } => openai::probe_models(base_url, api_key_env, false).await,
    }
}

/// Checks that the speech-synthesis server in `config` is reachable
/// (`GET {base_url}/models`).
///
/// Waits at most [`PROBE_TIMEOUT`].
///
/// # Errors
///
/// As for [`probe_stt`].
pub async fn probe_tts(config: &TtsConfig) -> Result<()> {
    match config {
        TtsConfig::OpenAi {
            base_url,
            api_key_env,
            ..
        } => openai::probe_models(base_url, api_key_env, true).await,
    }
}
