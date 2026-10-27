# Local inference stack (optional)

`compose.yml` runs OpenAI-compatible model servers in containers, for clients
that want ROCm, CUDA or Vulkan acceleration without linking a vendor GPU
runtime into the `voicelink` binary (invariant I3) and with the concierge LLM
as an HTTP server (invariant I4). `voicelink setup` offers it as the
"compose stack" LLM provider; STT and TTS can be pointed at it with the
"HTTP" providers.

| Service | Address | Image | Serves |
|---|---|---|---|
| `llm-*` | `http://127.0.0.1:8180/v1` | `ghcr.io/ggml-org/llama.cpp:server[-rocm\|-cuda\|-vulkan]` | `/v1/chat/completions`, started with `--jinja` so tool calls come back structured |
| `stt-*` | `http://127.0.0.1:8181/v1` | `ghcr.io/ggml-org/whisper.cpp:main[-cuda\|-vulkan]` | `whisper-server` on `/v1/audio/transcriptions` |
| `tts-*` | `http://127.0.0.1:8182/v1` | `ghcr.io/remsky/kokoro-fastapi-{cpu,gpu}` | `/v1/audio/speech` (Kokoro-82M is inside the image) |

Every port is published on `127.0.0.1` only.

## Use

Choose **one** profile; they all publish the same ports.

```sh
export VOICELINK_MODELS_DIR="$HOME/.local/share/voicelink/models"   # the default
export LLM_MODEL_FILE=my-model.gguf                                  # a GGUF you placed in that directory
export WHISPER_MODEL_FILE=ggml-large-v3-turbo-q8_0.bin               # fetched by `voicelink setup`
docker compose -f compose/compose.yml --profile vulkan up -d
```

| Profile | llama.cpp | whisper.cpp | Kokoro | Device access |
|---|---|---|---|---|
| `rocm` | `server-rocm` | `main-vulkan` | `kokoro-fastapi-cpu` | `/dev/kfd`, `/dev/dri`, groups `video` and `render` |
| `cuda` | `server-cuda` | `main-cuda` | `kokoro-fastapi-gpu` (`cu128`) | NVIDIA Container Toolkit (`deploy.resources.reservations.devices`) |
| `vulkan` | `server-vulkan` | `main-vulkan` | `kokoro-fastapi-cpu` | `/dev/dri`, groups `video` and `render` |
| `cpu` | `server` | `main` | `kokoro-fastapi-cpu` | none |

whisper.cpp publishes no ROCm image, so the `rocm` profile transcribes through
its Vulkan build (Mesa RADV). Kokoro-FastAPI's GPU image is CUDA only.

`group_add` takes numeric group ids. Find yours with
`getent group video render` and export `VIDEO_GID` and `RENDER_GID` if they
differ from the defaults (44 and 992).

Image tags can be overridden with `LLAMA_TAG_{ROCM,CUDA,VULKAN,CPU}`,
`WHISPER_TAG_{CUDA,VULKAN,CPU}`, `KOKORO_TAG_{GPU,CPU}`. The llama.cpp tags
float with upstream master; pin a `server-b<build>` tag once you have one you
trust.

## Pointing voicelink at it

```sh
voicelink setup --non-interactive --ssh machinelearn@192.168.2.132 --tier rocm-large \
  --llm compose --llm-model default \
  --stt http --stt-url http://127.0.0.1:8181/v1 \
  --tts http --tts-url http://127.0.0.1:8182/v1 --tts-voice af_heart
```

Then confirm the tool-calling probe in the wizard passes: if `llama-server`
returns plain text instead of `tool_calls`, the concierge cannot act.

## Notes

- The AMD Strix Halo APU is `gfx1151`. Whether the `server-rocm` image carries
  kernels for it depends on the upstream build; if it fails to start, use the
  `vulkan` profile, which does not depend on ROCm userspace.
- Do not run this stack inside the agentbox container: a Docker socket there
  talks to the host. It is for the laptop.
- Licences: llama.cpp and whisper.cpp are MIT, Kokoro-FastAPI is Apache-2.0,
  the Kokoro-82M weights are Apache-2.0.
