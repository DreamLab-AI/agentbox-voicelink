//! Interchangeable speech-to-text, chat-model and text-to-speech backends.
//!
//! Each capability is a trait ([`SpeechToText`], [`ChatModel`],
//! [`TextToSpeech`]) with implementations chosen at runtime from a serde
//! config enum ([`SttConfig`], [`LlmConfig`], [`TtsConfig`]):
//!
//! | Slot | Embedded (cargo feature) | Local or remote HTTP |
//! |---|---|---|
//! | STT | whisper.cpp (`whisper`, `whisper-vulkan`, `whisper-metal`) | OpenAI `/v1/audio/transcriptions`; Nemotron streaming WebSocket |
//! | LLM | — (always HTTP: a model server outlives the client) | OpenAI `/v1/chat/completions` with tool calls |
//! | TTS | — (see below) | OpenAI `/v1/audio/speech` (Kokoro-FastAPI, Pocket TTS) |
//!
//! ROCm and CUDA are reached through the HTTP implementations (a model server
//! built for that runtime), so client binaries never link a vendor GPU runtime.
//!
//! # Embedded Kokoro is not shipped
//!
//! Kokoro-82M needs a grapheme-to-phoneme front end, and every practical Rust
//! one links espeak-ng (GPL-3.0), which would break this crate's
//! `Apache-2.0 OR MIT` licence. [`TtsConfig::Kokoro`] therefore stays in the
//! config enum (so configuration files remain stable) but [`build_tts`] rejects
//! it with [`BackendError::Config`]; run Kokoro-FastAPI and select
//! [`TtsConfig::OpenAi`] instead.
//!
//! # Features
//!
//! | Feature | Effect |
//! |---|---|
//! | `whisper` | Embedded whisper.cpp, CPU (needs `cmake` and a C++ compiler to build) |
//! | `whisper-vulkan` | `whisper` plus the Vulkan GPU backend |
//! | `whisper-metal` | `whisper` plus the Metal GPU backend |
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
//! // An embedded backend whose feature is not compiled in is a clear error.
//! # #[cfg(not(feature = "whisper"))]
//! assert!(build_stt(&SttConfig::Whisper {
//!     model_path: "ggml-base.en.bin".into(),
//!     language: Some("en".into()),
//!     threads: 4,
//!     gpu: false,
//! })
//! .is_err());
//! ```

mod audio;
mod nemotron;
mod openai;
#[cfg(feature = "whisper")]
mod whisper;

pub use audio::{decode_wav, encode_wav, resample};

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
    /// The backend is misconfigured (missing model file, bad URL, feature not compiled in).
    #[error("configuration: {0}")]
    Config(String),
    /// Transport failure talking to a server.
    #[error("transport: {0}")]
    Transport(String),
    /// The server or engine answered with an error or an unparseable reply.
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
    /// whisper.cpp in-process (needs a `whisper*` feature).
    Whisper {
        /// Path to a ggml model file.
        model_path: String,
        /// Language code, or `None` to auto-detect.
        language: Option<String>,
        /// CPU threads.
        threads: u16,
        /// Use the GPU backend compiled in (Vulkan or Metal).
        gpu: bool,
    },
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
    /// Kokoro-82M in-process (needs the `kokoro` feature).
    Kokoro {
        /// Path to the Kokoro ONNX model.
        model_path: String,
        /// Path to the voices file.
        voices_path: String,
        /// Voice name.
        voice: String,
    },
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
/// [`BackendError::Config`] when the selected backend's feature is not
/// compiled in or its settings are invalid.
pub fn build_stt(config: &SttConfig) -> Result<Box<dyn SpeechToText>> {
    match config {
        #[cfg(feature = "whisper")]
        SttConfig::Whisper {
            model_path,
            language,
            threads,
            gpu,
        } => Ok(Box::new(whisper::WhisperStt::new(
            model_path, language, *threads, *gpu,
        )?)),
        #[cfg(not(feature = "whisper"))]
        SttConfig::Whisper { .. } => Err(BackendError::Config(
            "embedded whisper.cpp is not compiled in: enable the `whisper`, `whisper-vulkan` or `whisper-metal` feature".into(),
        )),
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
/// [`BackendError::Config`] when the selected backend is not available or its
/// settings are invalid. [`TtsConfig::Kokoro`] is always rejected: see the
/// crate documentation.
pub fn build_tts(config: &TtsConfig) -> Result<Box<dyn TextToSpeech>> {
    match config {
        TtsConfig::Kokoro { .. } => Err(BackendError::Config(
            "embedded Kokoro is not available (its phonemiser needs GPL espeak-ng); run Kokoro-FastAPI and use the `open_ai` TTS backend".into(),
        )),
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
