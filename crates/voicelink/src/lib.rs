//! The `voicelink` client as a library, so integration tests can drive the
//! real controller link ([`link`]) and the wizard's handshake
//! ([`setup::remote`]) against a real `voice-paned`. The `voicelink` binary is
//! a thin command-line shell over these modules; nothing here is a published
//! API.

#![allow(missing_docs)]

mod audio;
mod concierge;
pub mod config;
pub mod connect;
pub mod daemon;
pub mod link;
mod paths;
pub mod setup;
mod status;
pub mod status_cmd;
