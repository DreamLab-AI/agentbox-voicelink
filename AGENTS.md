# agentbox-voicelink — agent instructions

Canonical, tool-neutral rules for every coding agent working here. Agentbox side of the contract: `DreamLab-AI/agentbox` ADR-2151.

## Shape

| Crate | Published | Owns |
|---|---|---|
| `voicelink-proto` | crates.io | The JSON-lines wire contract and spool record. Additive changes (new message, enum value, optional field) keep `PROTOCOL_VERSION`; removals or meaning changes bump it. Decoders ignore unknown fields and fall back to `Unknown` variants (see the crate docs). |
| `tmux-control` | crates.io | Generic tmux control-mode client. No voicelink or agentbox concepts. |
| `voicelink-backends` | crates.io | `SpeechToText` / `ChatModel` / `TextToSpeech` traits and implementations. No audio I/O, no tmux. |
| `voice-paned` | no (baked into agentbox from a pinned rev) | Controller: grouped view session, focus following, focus-only writes, spool tailing. |
| `voicelink` | no (GitHub release binaries) | Client: onboarding wizard, launcher, audio pipeline, concierge, daemon. |

## Invariants

- **I1 Focus-only writes.** Nothing that types into a pane takes a pane argument; writes go to the view session's focused pane. Background action means `SelectWindow` first.
- **I2 No new ingress.** The only transport is the operator's ssh login (ControlMaster). Remote backends are `ssh -L` forwards on it.
- **I3 No vendor GPU runtime in client binaries.** Embedded builds are CPU, Vulkan (Linux) or Metal (macOS). ROCm/CUDA are reached through HTTP model servers.
- **I4 The LLM is always HTTP.** Only STT and TTS may be embedded.
- **I5 The daemon never writes to the attached terminal.** Logs go to a file; state goes to the status file.
- **I6 Onboarding completes before attach.** The wizard owns the terminal only while tmux does not.
- **I7 Published crates** ship full rustdoc (crate-level `//!`, every public item, compiling examples), a README and licence metadata, and pass `cargo doc --no-deps` with zero warnings.

## Build & test

```
export CARGO_TARGET_DIR=/home/devuser/workspace/.tmp/vl-target-<worker>   # in the agentbox container: one per worktree, outside the tree
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p voicelink-proto -p tmux-control -p voicelink-backends
```

Audio I/O (`cpal`, the `audio-io` feature, on by default) needs ALSA headers on Linux (`libasound2-dev`); CI installs them. Where they are missing, build and test the client with `--no-default-features`.

## Rules

- **Never touch a live tmux server.** Every test and dev run uses a private server (`tmux -L voicelink-test-<pid>` or `-S <tmpdir>/sock`) that the test starts and kills itself. `voice-paned` and `tmux-control` take the socket as a parameter for exactly this reason. In the agentbox container the default server is the operator's: a stray `send-keys` types into a live agent prompt.

- One checkout per worker: each parallel agent works in its own `git worktree`.
- Never hand-roll cryptography; this repo should need none.
- Licence: Apache-2.0 OR MIT. Never depend on an AGPL crate.
- UK English in docs and messages.
