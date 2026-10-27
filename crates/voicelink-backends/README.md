# voicelink-backends

Interchangeable speech-to-text, tool-calling chat and text-to-speech backends
behind three small async traits, chosen at runtime from a serde config enum.
No audio device I/O, no terminal handling: it turns audio into text, text into
a model reply (with tool calls) and text into audio.

| Slot | Trait | Backends |
|---|---|---|
| STT | `SpeechToText` | embedded whisper.cpp (`whisper*` features); OpenAI `/v1/audio/transcriptions`; Nemotron streaming ASR (WebSocket + MessagePack) |
| Chat | `ChatModel` | OpenAI `/v1/chat/completions` with tool calls (llama-server, Ollama, LM Studio, gateways); extra top-level request fields via `extra` |
| TTS | `TextToSpeech` | embedded Kokoro-82M (`kokoro`); OpenAI `/v1/audio/speech` with streamed PCM (Kokoro-FastAPI, Pocket TTS), WAV fallback |

## Cargo features

| Feature | Default | Effect |
|---|---|---|
| `whisper` | off | Embedded whisper.cpp on the CPU via `whisper-rs`. Needs `cmake`, a C++ compiler and libclang at build time. |
| `kokoro` | off | Embedded Kokoro-82M TTS (CPU) via ONNX Runtime. See below. |
| `whisper-vulkan` | off | `whisper` plus the Vulkan GPU backend. |
| `whisper-metal` | off | `whisper` plus the Metal GPU backend. |

HTTP and WebSocket backends are always available (`reqwest` and
`tokio-tungstenite` with rustls; no OpenSSL). CUDA and ROCm are deliberately
not linked: reach them through an HTTP model server built for that runtime.

Without a `whisper*` feature, `build_stt` returns `BackendError::Config` for
`SttConfig::Whisper`.

## Embedded Kokoro

The `kokoro` feature runs Kokoro-82M in-process through ONNX Runtime (`ort`),
with the ONNX model and voices file (`.npz`) from the
[kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases).
Phonemisation is `misaki-rs` with its `espeak` feature **off**: espeak-ng is
GPL-3.0 and would taint this crate's `Apache-2.0 OR MIT` licence. The
consequence is that words absent from the bundled dictionaries are spelled
letter by letter; for full-coverage pronunciation run Kokoro-FastAPI and use
`TtsConfig::OpenAi`. The feature downloads a prebuilt ONNX Runtime at build time
and needs Rust 1.92. ONNX Runtime allows one environment per process, so it
cannot share a process with other code that initialises `ort` itself.

## Example

```rust
use voicelink_backends::{build_llm, LlmConfig};

fn main() -> Result<(), voicelink_backends::BackendError> {
    let llm = build_llm(&LlmConfig::OpenAi {
        base_url: "http://127.0.0.1:8080/v1".into(),
        model: "local".into(),
        api_key_env: None,
        extra: serde_json::Map::new(),
    })?;
    println!("{}", llm.name());
    Ok(())
}
```

Streaming speech stops promptly with `BackendError::Cancelled` when the
receiving end of the sink channel is dropped, which is how a barge-in works.

## Helpers

`resample` (band-limited, via `rubato`), `encode_wav` and `decode_wav`.

## Licence

Licensed under either of Apache-2.0 or MIT at your option.
