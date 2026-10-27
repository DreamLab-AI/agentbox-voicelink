# voicelink-proto

Wire contract between a voice client and a tmux pane controller, used by
[agentbox-voicelink](https://github.com/DreamLab-AI/agentbox-voicelink).

- **JSON Lines**, one tagged object per line, both directions, over any byte
  stream (typically `ssh -T host … voice-paned --stdio`).
- **Versioned handshake**: `Hello { proto }` → `Welcome` or `ProtocolMismatch`.
- **Focus-only writes**: `Send`, `Keys` and `Interrupt` have no pane argument;
  they act on the pane focused in the client's own grouped tmux session.
  Reads (`ReadPane`, `ListPanes`) may address any pane.
- **Event spool**: `SpoolRecord` is the line format agent hooks append and
  the controller relays as `Event`.

```rust
use voicelink_proto::{ClientMessage, PROTOCOL_VERSION, encode_line};

let line = encode_line(&ClientMessage::Hello {
    proto: PROTOCOL_VERSION,
    client: "voicelink/0.1.0".into(),
    view_session: "remote-laptop".into(),
}).unwrap();
assert_eq!(
    line,
    "{\"type\":\"hello\",\"proto\":1,\"client\":\"voicelink/0.1.0\",\"view_session\":\"remote-laptop\"}\n"
);
```

## Licence

Licensed under either of [Apache-2.0](../../LICENSE-APACHE) or
[MIT](../../LICENSE-MIT) at your option.
