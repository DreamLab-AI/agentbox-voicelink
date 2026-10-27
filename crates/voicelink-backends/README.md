# voicelink-backends

Speech-to-text, tool-calling chat and text-to-speech clients behind three small
async traits, chosen at runtime from a serde config enum. Everything is reached
over HTTP or WebSocket: there are no embedded inference engines, so GPU
runtimes (ROCm, CUDA, Vulkan, Metal) stay in the model server and never in the
client binary. No audio device I/O, no terminal handling.

| Slot | Trait | Backends |
|---|---|---|
| STT | `SpeechToText` | OpenAI `/v1/audio/transcriptions` (whisper.cpp `whisper-server`, speaches, OpenAI); Nemotron streaming ASR (WebSocket + MessagePack, 30 s utterance guard) |
| Chat | `ChatModel` | OpenAI `/v1/chat/completions` with tool calls (llama-server, Ollama, LM Studio, gateways); extra top-level request fields via `extra` |
| TTS | `TextToSpeech` | OpenAI `/v1/audio/speech` with streamed PCM (Kokoro-FastAPI, Pocket TTS), WAV fallback |

## Features

None. HTTP and WebSocket use `reqwest` and `tokio-tungstenite` with rustls (no
OpenSSL). The minimum supported Rust version is 1.85 (derived from the edition
and the dependency set; CI builds with the current stable toolchain).

## Health checks

`probe_stt`, `probe_llm` and `probe_tts` make one cheap reachability check,
bounded by `PROBE_TIMEOUT` (3 s), and name the URL in every error:

- OpenAI-style servers: `GET {base_url}/models`. 2xx, 404 and 405 count as up
  (several speech servers have no model listing); 401/403 and 5xx are errors;
  a refused or timed-out connection is a `Transport` error.
- Nemotron: open the WebSocket, wait for the server's first frame (`Ready`, or
  `Error` when another client is streaming, which still proves it is alive) and
  close.

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
