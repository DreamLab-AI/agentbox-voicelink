# agentbox-voicelink

Talk to the coding agents in your agentbox. `voicelink` runs on your laptop,
listens to you, speaks back, and drives the agents running in agentbox's
shared tmux session: it types what you say into the focused pane, reads
permission prompts aloud, and tells you when an agent in another tab finishes.

Every model is reached over HTTP or WebSocket. There are no embedded inference
engines in the client, so a release is three plain binaries and the GPU stack
(ROCm, CUDA, Vulkan, Metal) is whatever your model servers use.

```
        laptop                                       agentbox host
 ┌─────────────────────────────┐   one ssh login  ┌──────────────────────────────┐
 │ mic ─► VAD/AEC ─► STT ──┐   │  (ControlMaster) │  docker exec ─► tmux session │
 │                         ▼   │ ═══════════════► │   ├─ voice-paned (controller)│
 │ speaker ◄─ TTS ◄─ concierge │  JSON lines over │   ├─ agent pane (claude)     │
 │                    (LLM)    │  stdio, plus     │   └─ agent pane (codex)      │
 │  STT / LLM / TTS servers:   │  ssh -L forwards │                              │
 │   compose stack, local, or ─┼─────────────────►│  Nemotron :8897, Loom :8084, │
 │   agentbox services         │   127.0.0.1 only │  Pocket TTS :8898            │
 └─────────────────────────────┘                  └──────────────────────────────┘
```

## Three ways to provide models

Each of the three model slots (speech-to-text, the concierge LLM, text-to-speech)
chooses its source independently in `voicelink setup`:

| Source | What it is | Good for |
|---|---|---|
| **Compose stack** | The bundled `compose/compose.yml`: llama.cpp, whisper.cpp `whisper-server` and Kokoro-FastAPI in containers, one profile each for `rocm`, `cuda`, `vulkan` and `cpu`, ports on `127.0.0.1` only, images pinned by digest. The wizard shows the exact `docker compose -f <path> --profile <p> up -d` command and runs it only when you confirm (`--start-stack` for headless setup). | Linux workstations with a GPU, or a CPU-only fallback |
| **Server already running** | Anything OpenAI-compatible on this machine. The wizard detects llama-server (`:8080`), Ollama (`:11434`), LM Studio (`:1234`), whisper-server and Kokoro-FastAPI (`:8880`). | macOS (Docker has no GPU there), or setups you already have |
| **agentbox services** | Nemotron streaming speech-to-text, the Ontology Loom and Pocket TTS on the agentbox host, over `ssh -L` forwards on the same login. | Thin clients, or when the agentbox host has the GPU |

## Quick start

1. Download the tarball for your machine from the
   [releases](https://github.com/DreamLab-AI/agentbox-voicelink/releases):
   `voicelink-x86_64-unknown-linux-gnu`, `voicelink-aarch64-unknown-linux-gnu`
   or `voicelink-aarch64-apple-darwin`. Verify the `.sha256`, unpack it and keep
   `compose/` next to the binary. On macOS, clear the quarantine flag
   (`xattr -d com.apple.quarantine voicelink agentbox-remote`): the binaries are
   unsigned.
2. `voicelink setup`: the wizard probes your hardware, proposes a tier and a
   source per slot, checks your ssh login and the controller on agentbox, and
   runs a speech smoke test. Re-running it pre-fills from your config.
3. `voicelink connect`: opens the ssh connection, probes every configured model
   endpoint (stopping with the compose command, or the service to start, for any
   that is down; `--skip-health` overrides), starts the voice daemon and attaches
   your terminal to the agentbox tmux session.

Linux builds need ALSA at run time. `voicelink status` shows the daemon's state.

### Build from source

Needs Rust 1.88 or newer, a C toolchain (for `ring`), `pkg-config` and the ALSA
headers. On Arch or CachyOS:

```sh
sudo pacman -S --needed base-devel git rustup alsa-lib pkgconf
rustup default stable
git clone https://github.com/DreamLab-AI/agentbox-voicelink
cd agentbox-voicelink
cargo build --release -p voicelink
./target/release/voicelink setup      # run from the checkout so compose/ is found
./target/release/voicelink connect
```

On Debian or Ubuntu the packages are `build-essential pkg-config libasound2-dev`.
`cargo install --path crates/voicelink` puts `voicelink` and `agentbox-remote`
in `~/.cargo/bin`; from there, set `VOICELINK_COMPOSE=/path/to/checkout/compose/compose.yml`
so the wizard can find the compose stack. The mute hotkey is X11/macOS only;
`--no-default-features --features audio-io` builds without it.

## Tiers

The tier decides the compose profile and the model sizes the wizard proposes. You
can change every choice.

| Tier | Detected as | Compose profile | Speech-to-text | Concierge LLM | Text-to-speech |
|---|---|---|---|---|---|
| `rocm-large` | ROCm GPU with 32 GiB or more unified or dedicated memory | `rocm`, or `vulkan` on Strix Halo (gfx1151) | whisper large-v3-turbo | 27B-35B tool-calling model at Q4 | Kokoro-FastAPI |
| `nvidia` | NVIDIA with 12 GiB or more | `cuda` | whisper medium q8 | 8B-14B at Q4 | Kokoro-FastAPI (GPU) |
| `small-gpu` | any other GPU | `vulkan` | whisper small q8 | 4B-8B at Q4 | Kokoro-FastAPI |
| `metal` | Apple silicon | none: native servers (Homebrew whisper-cpp and llama.cpp, Ollama, LM Studio) | whisper large-v3-turbo q8 | 8B-14B via Ollama or LM Studio | agentbox Pocket TTS |
| `cpu` | no usable GPU | `cpu` | whisper base.en q8 | agentbox Loom | Kokoro-FastAPI (CPU) |
| `thin` | under 4 GiB RAM | none | agentbox Nemotron | agentbox Loom | agentbox Pocket TTS |

On aarch64 Linux the whisper.cpp images are amd64-only, so the wizard moves
speech-to-text to agentbox Nemotron (or a server you run) and says so.

The concierge model must return structured tool calls (llama-server needs
`--jinja`); the wizard's tool-calling probe checks. Details of the stack, the
image pins and the AMD Strix Halo recommendation are in
[compose/README.md](compose/README.md).

## The agentbox side

agentbox bakes the controller (`voice-paned`) into its image from a pinned
revision of this repository, enabled by the `[voicelink]` manifest slot
(ADR-2151 in [DreamLab-AI/agentbox](https://github.com/DreamLab-AI/agentbox)).
The client reaches it with `ssh <host> docker exec -i --user <uid> <container>
voice-paned --stdio --session <name>`; there is no new network ingress, only the login you already use.
Agent hooks (a Claude Code `Stop` hook, a Codex `notify` program) append to a
bounded event spool that the controller relays as events.

## Safety model

- **Focus-only writes.** Nothing that types into a pane takes a pane argument.
  Text, keys and interrupts act on the pane focused in your view session; acting
  on a background agent means moving focus there first, which you can see.
- **`PromptActive`.** While an agent shows a permission or choice prompt, the
  controller refuses to type free text, and `force` never overrides it. Prompts
  are answered with explicit key presses.
- **Read-only event turns.** When an agent finishes in another tab, the concierge
  summarises it in a turn that has read-only tools only and treats the agent's
  message as untrusted data. Event turns are not added to the conversation
  history, so they cannot steer a later write-capable turn.
- **One transport.** Everything rides the operator's ssh login (ControlMaster);
  remote model services are `ssh -L` forwards bound to `127.0.0.1`.
- **Terminal discipline.** The wizard runs before tmux takes the terminal, and the
  daemon never writes to it: logs go to a file, state to a status file.

## Crates

| Crate | Published | Role |
|---|---|---|
| [`voicelink-proto`](crates/voicelink-proto) | crates.io | JSON-lines wire contract and spool record |
| [`tmux-control`](crates/tmux-control) | crates.io | generic tmux control-mode client |
| [`voicelink-backends`](crates/voicelink-backends) | crates.io | STT, chat and TTS clients over HTTP and WebSocket, with health probes |
| `voice-paned` | no (baked into agentbox) | controller beside the tmux server |
| `voicelink` | GitHub release binaries | the client: wizard, launcher, audio, concierge, daemon |

Contributor rules and the invariants are in [AGENTS.md](AGENTS.md).

## Licence

Apache-2.0 OR MIT, at your option. Everything the compose stack runs keeps its
own licence: llama.cpp and whisper.cpp are MIT, Kokoro-FastAPI and the
Kokoro-82M weights are Apache-2.0.
