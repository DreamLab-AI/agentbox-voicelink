//! An asynchronous client for tmux **control mode** (`tmux -C`).
//!
//! Control mode turns tmux into a line protocol: you write commands to its
//! standard input, and it writes each command's output wrapped in
//! `%begin`/`%end` (or `%error`) guards, interleaved with asynchronous
//! `%notification` lines about windows, panes and sessions. This crate
//! implements the CONTROL MODE section of tmux(1) with Tokio:
//!
//! * [`Client`] attaches to a session and correlates every
//!   [`command`](Client::command) with its reply;
//! * [`Notification`] is a typed enum of the asynchronous events, delivered on
//!   a channel;
//! * [`Decoder`] is the pure parser behind it, usable on its own;
//! * [`quote`], [`escape_format`] and [`command_line`] build command lines from
//!   untrusted strings without injection;
//! * [`run`] and [`ensure_grouped_session`] cover the one-shot cases.
//!
//! Every API takes a [`Socket`], so a program can address its own private tmux
//! server (`-L name` or `-S path`) instead of the user's default one.
//!
//! # Example
//!
//! ```no_run
//! use tmux_control::{command_line, Client, ClientOptions, Notification, Socket};
//!
//! # async fn demo() -> Result<(), tmux_control::Error> {
//! let socket = Socket::Name("work".into());
//! let options = ClientOptions::new("main").flag("ignore-size").flag("no-output");
//! let (client, mut events) = Client::attach(&socket, &options)?;
//!
//! // Commands return their output lines (or tmux's error text).
//! let panes = client.command("list-panes -s -F '#{pane_id}'").await?;
//! println!("{panes:?}");
//!
//! // Untrusted text is quoted into a single inert word.
//! let hostile = "x; kill-server";
//! client.command(command_line(["rename-window", "-t", "@1", hostile])).await?;
//!
//! // Notifications arrive on a channel.
//! while let Some(event) = events.recv().await {
//!     if let Notification::WindowPaneChanged { window, pane } = event {
//!         println!("{window} now has active pane {pane}");
//!     }
//! }
//! client.detach().await;
//! # Ok(()) }
//! ```
//!
//! # Protocol notes
//!
//! * Only the `%end`/`%error` carrying the same command number closes a
//!   `%begin` block. Everything inside it, even a line starting with `%`
//!   (pane ids do), is output. tmux queues notifications until a block ends.
//! * Replies are matched to commands in order, using the block flag that marks
//!   a reply as answering a command this client sent.
//! * An empty line detaches a control client, so [`Client::command`] refuses
//!   empty commands and raw line breaks.
//! * `%output` payloads are returned as bytes after octal-unescaping; they are
//!   not guaranteed to be valid UTF-8.
//! * Control-mode clients that do not use the `ignore-size` flag take part in
//!   tmux's window-size calculation. Attach with `ignore-size` when the
//!   client is not a terminal, and never send `refresh-client -C`.
//! * Spawned `tmux` processes are killed if the [`Client`] is dropped while the
//!   runtime is shutting down; the tmux server is never affected.

mod client;
mod error;
mod oneshot;
mod parser;
mod quote;
mod socket;

pub use client::{Client, ClientOptions, Notifications};
pub use error::Error;
pub use oneshot::{ensure_grouped_session, has_session, run};
pub use parser::{Decoder, Event, Notification, Reply, parse_notification, unescape_output};
pub use quote::{command_line, escape_format, quote};
pub use socket::Socket;
