# Local inference stack

`compose.yml` runs OpenAI-compatible model servers in containers. voicelink
reaches every model over HTTP, so ROCm, CUDA and Vulkan live in these images
and never in the `voicelink` binary. It is one of three ways to provide models
(the others are servers you already run, and the agentbox services over the ssh
forwards). `voicelink setup` shows the exact command for your choices and runs
it only when you confirm; headless, `voicelink setup --non-interactive
--start-stack` does the same.

The release tarball carries this directory next to the binary
(`voicelink`, `compose/compose.yml`, `compose/README.md`); the wizard looks
there first, then in `../share/voicelink/compose/`, then in `./compose/`, and
`VOICELINK_COMPOSE=/path/to/compose.yml` overrides all three.

| Service | Address | Image | Serves |
|---|---|---|---|
| `llm-*` | `http://127.0.0.1:8180/v1` | `ghcr.io/ggml-org/llama.cpp:server[-rocm\|-cuda\|-vulkan]` | `/v1/chat/completions`, started with `--jinja` so tool calls come back structured |
| `stt-*` | `http://127.0.0.1:8181/v1` | `ghcr.io/ggml-org/whisper.cpp:main[-cuda\|-vulkan]` | `whisper-server` on `/v1/audio/transcriptions` |
| `tts-*` | `http://127.0.0.1:8182/v1` | `ghcr.io/remsky/kokoro-fastapi-{cpu,gpu}` | `/v1/audio/speech` (the Kokoro-82M weights are inside the image) |

Every port is published on `127.0.0.1` only. Every service has a healthcheck
(`curl` is present in all nine images): `/health` on llama-server,
whisper-server and Kokoro-FastAPI. `docker compose ps` shows `healthy` once the
model is loaded.

## Use

Choose **one** profile; they all publish the same ports.

```sh
export VOICELINK_MODELS_DIR="$HOME/.local/share/voicelink/models"   # default; the wizard downloads here
export LLM_MODEL_FILE=my-model.gguf                                  # a tool-calling GGUF you placed in that directory
export WHISPER_MODEL_FILE=ggml-large-v3-turbo-q8_0.bin               # fetched by `voicelink setup`
docker compose -f compose/compose.yml --profile vulkan up -d stt-vulkan tts-cpu llm-vulkan
```

Naming the services starts only the slots you serve from the stack. Without a
GGUF in the models directory, leave `llm-*` out: it would restart in a loop.

The models directory is shared: the wizard fetches the whisper model
(SHA-256 verified) into the `models_dir` of your config and passes that same
directory to compose as `VOICELINK_MODELS_DIR`. The LLM GGUF is not in the
wizard's catalogue because the model is your choice and must support tool
calls; the wizard prints a size guide per tier.

| Profile | llama.cpp | whisper.cpp | Kokoro | Device access |
|---|---|---|---|---|
| `rocm` | `server-rocm` | `main-vulkan` | `kokoro-fastapi-cpu` | `/dev/kfd`, `/dev/dri`, groups `video` and `render` |
| `cuda` | `server-cuda` | `main-cuda` | `kokoro-fastapi-gpu` (`cu128`) | NVIDIA Container Toolkit (`deploy.resources.reservations.devices`) |
| `vulkan` | `server-vulkan` | `main-vulkan` | `kokoro-fastapi-cpu` | `/dev/dri`, groups `video` and `render` |
| `cpu` | `server` | `main` | `kokoro-fastapi-cpu` | none |

`group_add` takes numeric group ids (defaults 44 and 992). The wizard reads
`/etc/group` and exports `VIDEO_GID` and `RENDER_GID` when yours differ;
otherwise find them with `getent group video render`.

## AMD Strix Halo (gfx1151): use `vulkan`

`vulkan` is the recommended profile for the Ryzen AI Max / Strix Halo APU, and
the wizard proposes it when `rocminfo` reports `gfx1151`. The reasons, and what
is not known:

- whisper.cpp publishes no ROCm image, so the `rocm` profile already runs its
  speech service on Vulkan (Mesa RADV). The `vulkan` profile puts the LLM on the
  same driver stack, so the three services share one GPU path and nothing
  depends on the host's ROCm userspace or kernel module versions.
- llama.cpp's `server-rocm` image does list `gfx1151` in its build targets
  (`ROCM_DOCKER_ARCH` in `.devops/rocm.Dockerfile` at revision 7aca0d2, the
  build pinned here), so `rocm` can work where the host kernel and `amdgpu`
  driver support the chip. It is the fragile path, not an unsupported one.
- Not measured: no throughput comparison between the two profiles was run for
  this README. Switch to `rocm` if you want to benchmark it yourself.

## Image pins

Every image is pinned as `repo:tag@sha256:<digest>`; the tag documents intent
and the digest is what is pulled. The digests are those of the multi-arch
image indexes, resolved from the registry on 2026-10-27 with an anonymous
`ghcr.io/token` bearer token and a manifest request carrying
`Accept: application/vnd.oci.image.index.v1+json`, reading the
`Docker-Content-Digest` response header:

| Image | Platforms | Digest |
|---|---|---|
| `ghcr.io/ggml-org/llama.cpp:server-rocm` | linux/amd64 | `sha256:31b19613c7ae55048dd113a3f4e7d093308eb9b2370a02d4d06782102bd6ea2c` |
| `ghcr.io/ggml-org/llama.cpp:server-cuda` | linux/amd64, linux/arm64 | `sha256:8c219fe0367f7e794e3c69da50c5992d474077d60794a218b9e9a3b0c5aed688` |
| `ghcr.io/ggml-org/llama.cpp:server-vulkan` | linux/amd64, linux/arm64 | `sha256:3607a208ef03bc4f64515fbfac6bffc386859c14b64abec24c96f143868fcb37` |
| `ghcr.io/ggml-org/llama.cpp:server` | linux/amd64, linux/arm64, linux/s390x | `sha256:8ee4ec7ce58a921313d07390c6699af13aa5ec083f3cdf67e5f4155767047373` |
| `ghcr.io/ggml-org/whisper.cpp:main` | linux/amd64 | `sha256:cdfceef9caa5fed3efaf07bfc7d1ea0e43d28536e1521f448570b0ab02886abb` |
| `ghcr.io/ggml-org/whisper.cpp:main-vulkan` | linux/amd64 | `sha256:a1d539e66d62a0e9d655237b94bc0e91cad5fbcb54c2bfb57d9b830e7fec378f` |
| `ghcr.io/ggml-org/whisper.cpp:main-cuda` | linux/amd64 | `sha256:b0d3a90a4db2f975391614c65e53f2ecf63f28dbfe1329ea53b5154af8bd42ed` |
| `ghcr.io/remsky/kokoro-fastapi-cpu:v0.9.0` | linux/amd64, linux/arm64 | `sha256:ee3111d6a2c903ed62f3b4fa19543c6901205ed39fce3c177e895b34a8386b9c` |
| `ghcr.io/remsky/kokoro-fastapi-gpu:v0.9.0-cu128` | linux/amd64 | `sha256:c6683ee24a3b3bc3b49130ecc52c32e8010a1052ae32781a91682fa09c6ba984` |

**The whisper.cpp images are `linux/amd64` only**, as are `server-rocm` and the
Kokoro GPU image (read from the pinned image indexes). On an aarch64 Linux host
(for example the `voicelink-aarch64-unknown-linux-gnu` binary on a Raspberry Pi
or an Ampere workstation) the stack cannot serve speech-to-text: the wizard
proposes the agentbox Nemotron service for that slot and says why, and warns
before it runs a stack command that names an image missing for your
architecture. The Kokoro CPU image and the llama.cpp `server`, `server-vulkan`
and `server-cuda` images do have `linux/arm64`. macOS hosts run native servers
instead of Docker.

The llama.cpp tags float with upstream master; at pin time all four were
build b11913 (revision `7aca0d2ea8d9`). There is deliberately no tag override
variable: a digest wins over a tag, so an override would not do what it says.
To update, resolve the new digest the same way and edit `compose.yml`; a test in
the `voicelink` crate fails if any image loses its digest.

### What was confirmed in the images

Inspected through the registry API (image config and layer listings), without
running anything:

- `whisper-server` exists in all three whisper.cpp images: `/app/build/bin/` in
  `main` and `main-cuda` (which set `PATH` and `ENTRYPOINT ["bash","-c"]`), and
  `/usr/local/bin/` in `main-vulkan` (no entrypoint, so `compose.yml` sets
  `entrypoint: [bash, -c]`). The binary in `main-vulkan` lists `--inference-path`,
  `--host` and a `/health` route; `main` and `main-cuda` were not unpacked, so
  those flags are assumed to match (same upstream build).
- `curl` is installed in all nine images; llama.cpp images declare
  `HEALTHCHECK curl -f http://localhost:8080/health` themselves.
- Kokoro-FastAPI downloads its weights at image build time
  (`DOWNLOAD_MODEL=true`), so it needs no files from the models directory; its
  source serves `/health` and `/v1/models` on port 8880.
- `main-cuda` sets `NVIDIA_REQUIRE_CUDA=cuda>=13.0` (CUDA 13.0 runtime) and
  Kokoro's GPU image needs CUDA 12.8: the NVIDIA driver must be recent enough
  for both.

Not verified: nothing was started, so GPU access, model loading and the
healthchecks have not been exercised on hardware.

## Pointing voicelink at it

```sh
voicelink setup --non-interactive --ssh machinelearn@192.168.2.132 --tier nvidia \
  --stt compose --llm compose --llm-file my-model.gguf --tts compose
```

Then confirm the tool-calling probe in the wizard passes: if `llama-server`
returns plain text instead of `tool_calls`, the concierge cannot act. Before it
attaches, `voicelink connect` probes all three endpoints and prints the
compose command if the stack is down.

## Notes

- Do not run this stack inside the agentbox container: a Docker socket there
  talks to the host. It is for the laptop or workstation.
- Licences: llama.cpp and whisper.cpp are MIT, Kokoro-FastAPI is Apache-2.0,
  the Kokoro-82M weights are Apache-2.0.
