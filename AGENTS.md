# agentbox-voicelink — agent instructions

Canonical, tool-neutral rules for every coding agent working here. Agentbox side of the contract: `DreamLab-AI/agentbox` ADR-2151.

## Shape

| Crate | Published | Owns |
|---|---|---|
| `voicelink-proto` | crates.io | The JSON-lines wire contract and spool record. Additive changes (new message, enum value, optional field) keep `PROTOCOL_VERSION`; removals or meaning changes bump it. Decoders ignore unknown fields and fall back to `Unknown` variants (see the crate docs). |
| `tmux-control` | crates.io | Generic tmux control-mode client. No voicelink or agentbox concepts. |
| `voicelink-backends` | crates.io | `SpeechToText` / `ChatModel` / `TextToSpeech` traits and their HTTP and WebSocket implementations, plus `probe_*` health checks. No inference engines, no audio I/O, no tmux. Declares its own `rust-version` (1.85). |
| `voice-paned` | no (baked into agentbox from a pinned rev) | Controller: grouped view session, focus following, focus-only writes, spool tailing. |
| `voicelink` | no (GitHub release binaries) | Client: onboarding wizard (compose stack / local servers / agentbox services per slot), launcher with a pre-attach health gate, audio pipeline, concierge, daemon. Feature `hotkey` (default on) gates `global-hotkey`. |
| `compose/` | shipped inside each release tarball | Optional Docker model stack (llama.cpp, whisper.cpp, Kokoro-FastAPI), images pinned by digest, loopback ports, one shared models directory. |

## Invariants

- **I1 Focus-only writes.** Nothing that types into a pane takes a pane argument; writes go to the view session's focused pane. Background action means `SelectWindow` first.
- **I2 No new ingress.** The only transport is the operator's ssh login (ControlMaster). Remote backends are `ssh -L` forwards on it.
- **I3 No inference engine in any binary.** The client links no whisper.cpp, ONNX Runtime, llama.cpp or vendor GPU runtime (ROCm, CUDA, Vulkan, Metal); GPUs belong to the model servers. Release binaries are plain default-feature builds.
- **I4 Every model is reached over HTTP or WebSocket.** STT, LLM and TTS are OpenAI-compatible HTTP servers or the Nemotron WebSocket, from the compose stack, a server already running, or an agentbox service over the ssh forwards. A config that selects an embedded engine is rejected with advice to re-run `voicelink setup`, which migrates it (ssh, audio, hotkey and LLM settings kept, speech slots chosen again).
- **I5 The daemon never writes to the attached terminal.** Logs go to a file; state goes to the status file.
- **I6 Onboarding completes before attach.** The wizard owns the terminal only while tmux does not.
- **I7 Published crates** ship full rustdoc (crate-level `//!`, every public item, compiling examples), a README and licence metadata, and pass `cargo doc --no-deps` with zero warnings.

## Docker and compose

- The only code that runs `docker` is the setup effect runner (and the headless `--start-stack` path), and only after an explicit confirmation in the UI or the flag. Tests assert the emitted `Effect::StartStack` and never run it; never run `docker compose` from a worker here, the socket in the agentbox container is the host's.
- Every image in `compose/compose.yml` is `repo:tag@sha256:<digest>` resolved from the registry (never invented), ports bind `127.0.0.1`, every service has a healthcheck; `stack.rs` has a test for all three.
- `voicelink connect` probes all three endpoints through the forwards before it hands the terminal to tmux; `--skip-health` overrides.

## Build & test

```
export CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/vl-target-<worker>   # in the agentbox container: one per worktree, outside the tree
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
# the agentbox Nix build runs voice-paned's tests on Linux without X11 or ALSA: keep that closure free of them
cargo tree -p voice-paned --no-default-features -e normal,dev | grep -iE 'global-hotkey|x11|xkb'   # must print nothing
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p voicelink-proto -p tmux-control -p voicelink-backends
```

Audio I/O (`cpal`, the `audio-io` feature, on by default) needs ALSA headers on Linux (`libasound2-dev`); CI installs them. Where they are missing, build and test the client with `--no-default-features` (which also drops the `hotkey` feature and its X11 crates).

## Rules

- **Never touch a live tmux server.** Every test and dev run uses a private server (`tmux -L voicelink-test-<pid>` or `-S <tmpdir>/sock`) that the test starts and kills itself. `voice-paned` and `tmux-control` take the socket as a parameter for exactly this reason. In the agentbox container the default server is the operator's: a stray `send-keys` types into a live agent prompt.

- One checkout per worker: each parallel agent works in its own `git worktree`.
- Never hand-roll cryptography; this repo should need none.
- Licence: Apache-2.0 OR MIT. Never depend on an AGPL crate.
- UK English in docs and messages.
