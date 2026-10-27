# tmux-control

An asynchronous (Tokio) client for tmux **control mode** (`tmux -C`).

Control mode turns tmux into a line protocol: commands go to its standard
input, each command's output comes back wrapped in `%begin`/`%end` (or
`%error`) guards, and asynchronous `%notification` lines report what happens
to windows, panes and sessions. `tmux-control` implements the CONTROL MODE
section of tmux(1):

- `Client` attaches to a session and correlates each `command(..).await`
  with its reply (output lines, or tmux's error text).
- `Notification` is a typed enum of the asynchronous events (`%output` with
  octal-escape decoding, `%session-window-changed`, `%window-pane-changed`,
  `%layout-change`, `%exit`, `%subscription-changed`, ...), delivered on a
  channel. Anything unmodelled arrives as `Notification::Unknown`.
- `Decoder` is the pure, I/O-free parser, usable on its own.
- `quote`, `escape_format` and `command_line` build command lines from
  untrusted strings without injection (`;`, quotes, `$`, newlines, `#{`).
- `run` and `ensure_grouped_session` cover one-shot invocations and the
  idempotent, race-tolerant creation of a grouped (shared-windows) session.
- Every call takes a `Socket` (`Default`, `-L name`, `-S path`), so tools and
  tests can use a private server.

```rust,no_run
use tmux_control::{command_line, Client, ClientOptions, Socket};

#[tokio::main]
async fn main() -> Result<(), tmux_control::Error> {
let (client, mut events) = Client::attach(
    &Socket::Name("work".into()),
    &ClientOptions::new("main").flag("ignore-size").flag("no-output"),
)?;
let panes = client.command("list-panes -s -F '#{pane_id}'").await?;
client.command(command_line(["rename-window", "-t", "@1", "x; kill-server"])).await?;
while let Some(event) = events.recv().await {
    println!("{event:?}");
}
client.detach().await;
Ok(())
}
```

## Notes

- Only the `%end`/`%error` with the same command number closes a block;
  lines starting with `%` inside it (pane ids!) are output.
- An empty line detaches a control client, so empty commands and raw line
  breaks are refused locally.
- A control client without the `ignore-size` flag takes part in tmux's
  window-size calculation. Use `ignore-size` for non-terminal clients and never
  send `refresh-client -C`.
- `new-session -t` with an unknown target silently creates a new *group*;
  `ensure_grouped_session` therefore checks the target exists first.
- Names are matched exactly (`=name`), never by prefix.

Requires tmux 3.2 or newer (developed against 3.8).

## Licence

Apache-2.0 OR MIT.
