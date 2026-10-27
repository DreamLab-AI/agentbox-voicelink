# voicelink-backends

Interchangeable speech-to-text, tool-calling chat and text-to-speech backends
behind three small async traits, chosen at runtime from a serde config enum.
No audio device I/O, no terminal handling: it turns audio into text, text into
a model reply (with tool calls) and text into audio.

| Slot | Trait | Backends |
|---|---|---|
| STT | `SpeechToText` | embedded whisper.cpp (`whisper*` features); OpenAI `/v1/audio/transcriptions`; Nemotron streaming ASR (WebSocket + MessagePack) |
| Chat | `ChatModel` | OpenAI `/v1/chat/completions` with tool calls (llama-server, Ollama, LM Studio, gateways); extra top-level request fields via `extra` |
| TTS | `TextToSpeech` | OpenAI `/v1/audio/speech` with streamed PCM (Kokoro-FastAPI, Pocket TTS), WAV fallback |

## Cargo features

| Feature | Default | Effect |
|---|---|---|
| `whisper` | off | Embedded whisper.cpp on the CPU via `whisper-rs`. Needs `cmake`, a C++ compiler and libclang at build time. |
| `whisper-vulkan` | off | `whisper` plus the Vulkan GPU backend. |
| `whisper-metal` | off | `whisper` plus the Metal GPU backend. |

HTTP and WebSocket backends are always available (`reqwest` and
`tokio-tungstenite` with rustls; no OpenSSL). CUDA and ROCm are deliberately
not linked: reach them through an HTTP model server built for that runtime.

Without a `whisper*` feature, `build_stt` returns `BackendError::Config` for
`SttConfig::Whisper`.

## Embedded Kokoro

Not shipped. Kokoro-82M needs a phonemiser and every practical Rust one links
espeak-ng, which is GPL-3.0 and incompatible with this crate's
`Apache-2.0 OR MIT` licence. `TtsConfig::Kokoro` is kept so configuration files
stay stable, but `build_tts` rejects it; run Kokoro-FastAPI and use
`TtsConfig::OpenAi`.

## Example

```rust
use voicelink_backends::{build_llm, LlmConfig};

let llm = build_llm(&LlmConfig::OpenAi {
    base_url: "http://127.0.0.1:8080/v1".into(),
    model: "local".into(),
    api_key_env: None,
    extra: serde_json::Map::new(),
})?;
println!("{}", llm.name());
# Ok::<(), voicelink_backends::BackendError>(())
```

Streaming speech stops promptly with `BackendError::Cancelled` when the
receiving end of the sink channel is dropped, which is how a barge-in works.

## Helpers

`resample` (band-limited, via `rubato`), `encode_wav` and `decode_wav`.

## Licence

Licensed under either of Apache-2.0 or MIT at your option.
