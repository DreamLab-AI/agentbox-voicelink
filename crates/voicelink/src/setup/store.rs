//! Reading and atomically writing `config.toml`.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, bail};

use crate::config::Config;

/// Loads a config file.
pub fn load(path: &Path) -> anyhow::Result<Config> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text).with_context(|| format!("parsing {}", path.display()))
}

/// A config written by a voicelink that embedded its speech engines.
///
/// Those builds wrote `kind = "whisper"` under `[stt]` and `kind = "kokoro"`
/// under `[tts]` (`embedded` was only the command-line spelling). This build
/// reaches every model over HTTP, so such a file cannot be loaded.
#[derive(Debug, thiserror::Error)]
#[error(
    "this config selects an embedded speech engine ({section}.kind = {kind:?}), which this voicelink no longer includes: it reaches every model over HTTP. Run `voicelink setup` to choose a server (compose stack, a local server, or the agentbox services); your ssh, audio and hotkey settings are kept"
)]
pub struct LegacyEmbedded {
    /// `stt` or `tts`.
    pub section: &'static str,
    /// The tag found.
    pub kind: String,
}

/// Finds an embedded-engine tag in config text.
fn legacy_embedded(text: &str) -> Option<LegacyEmbedded> {
    let table: toml::Table = toml::from_str(text).ok()?;
    ["stt", "tts"].into_iter().find_map(|section| {
        let kind = table.get(section)?.get("kind")?.as_str()?;
        matches!(kind, "whisper" | "kokoro" | "embedded").then(|| LegacyEmbedded {
            section,
            kind: kind.to_string(),
        })
    })
}

/// Parses config text.
///
/// # Errors
///
/// [`LegacyEmbedded`] (downcastable) for a config from an embedded-engine
/// build; a TOML error otherwise.
pub fn parse(text: &str) -> anyhow::Result<Config> {
    match toml::from_str(text) {
        Ok(c) => Ok(c),
        Err(e) => match legacy_embedded(text) {
            Some(l) => Err(l.into()),
            None => Err(e.into()),
        },
    }
}

/// Rebuilds a usable config from a legacy one: the connection, audio, hotkey
/// concierge and LLM tables and the models directory are kept; the STT and TTS
/// tables (and any stack) take `defaults`.
/// `None` when the text is not a parseable legacy config.
pub fn migrate_legacy(text: &str, defaults: &Config) -> Option<Config> {
    legacy_embedded(text)?;
    let mut table: toml::Table = toml::from_str(text).ok()?;
    let fresh: toml::Table = toml::from_str(&toml::to_string(defaults).ok()?).ok()?;
    for key in ["stt", "tts", "stack"] {
        table.remove(key);
        if let Some(v) = fresh.get(key) {
            table.insert(key.to_string(), v.clone());
        }
    }
    toml::Value::Table(table).try_into().ok()
}

/// Serialises a config. TOML has no null, so JSON nulls in the LLM `extra`
/// map are rejected rather than silently dropped.
pub fn render(config: &Config) -> anyhow::Result<String> {
    let voicelink_backends::LlmConfig::OpenAi { extra, .. } = &config.llm;
    if extra.values().any(contains_null) {
        bail!("llm.extra must not contain null values (TOML cannot represent them)");
    }
    Ok(toml::to_string_pretty(config)?)
}

fn contains_null(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Null => true,
        serde_json::Value::Array(a) => a.iter().any(contains_null),
        serde_json::Value::Object(o) => o.values().any(contains_null),
        _ => false,
    }
}

/// Writes `config` to `path` atomically: a temporary file in the same
/// directory (mode 0600, fsynced) renamed over the target.
pub fn save(path: &Path, config: &Config) -> anyhow::Result<()> {
    let text = render(config)?;
    write_private(path, text.as_bytes())
}

/// Atomic 0600 write of arbitrary bytes.
pub fn write_private(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let dir = path
        .parent()
        .context("config path has no parent directory")?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let tmp = dir.join(format!(
        ".{}.tmp{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("config"),
        std::process::id()
    ));
    let result = (|| -> anyhow::Result<()> {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts
            .open(&tmp)
            .with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path).with_context(|| format!("renaming into {}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// What the embedded-engine builds wrote, taken from their serde shapes.
#[cfg(test)]
pub(crate) const LEGACY: &str = r#"
models_dir = "/home/me/.local/share/voicelink/models"

[connection]
ssh_destination = "me@box"
container = "agentbox"
container_user = "1000"
session = "agentbox"
view_session = "remote-laptop"
paned_command = "/opt/agentbox/bin/voice-paned"
forwards = []

[audio]
input_device = "USB Mic"
echo_cancel = true
vad_threshold = 0.5
end_silence_ms = 700
min_utterance_ms = 300
barge_in = true

[hotkey]
enabled = true
mute = "Ctrl+Shift+M"

[concierge]
speak_turn_end = true
announce_background = true
announce_actions = true
summary_words = 40
history_turns = 12

[stt]
kind = "whisper"
model_path = "/home/me/.local/share/voicelink/models/ggml-base.en-q8_0.bin"
language = "en"
threads = 4
gpu = false

[llm]
kind = "open_ai"
base_url = "http://127.0.0.1:8080/v1"
model = "default"

[tts]
kind = "kokoro"
model_path = "/m/kokoro-v1.0.int8.onnx"
voices_path = "/m/voices-v1.0.bin"
voice = "af_heart"
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::choices::{Choices, Source};

    #[test]
    fn round_trips_every_source_combination() {
        let mut c = Choices::defaults_for(crate::setup::choices::Tier::Cpu, "host");
        for stt in Source::ALL {
            for tts in Source::ALL {
                for llm in Source::ALL {
                    c.stt_source = *stt;
                    c.tts_source = *tts;
                    c.llm_source = *llm;
                    c.input_device = Some("Mic".into());
                    c.output_device = None;
                    let cfg = c.to_config();
                    let text = render(&cfg).unwrap();
                    assert_eq!(parse(&text).unwrap(), cfg, "{text}");
                }
            }
        }
    }

    #[test]
    fn embedded_configs_get_a_clear_error_not_a_parse_failure() {
        let e = parse(LEGACY).unwrap_err();
        let l = e.downcast_ref::<LegacyEmbedded>().expect("typed error");
        assert_eq!((l.section, l.kind.as_str()), ("stt", "whisper"));
        let msg = e.to_string();
        assert!(
            msg.contains("voicelink setup") && msg.contains("HTTP"),
            "{msg}"
        );
        // The command-line spelling and a kokoro-only file are caught too.
        let tts_only = LEGACY.replace(
            "kind = \"whisper\"",
            "kind = \"nemotron\"\nurl = \"ws://x/y\"",
        );
        let e = parse(&tts_only).unwrap_err();
        assert_eq!(e.downcast_ref::<LegacyEmbedded>().unwrap().section, "tts");
        let embedded = LEGACY.replace("\"whisper\"", "\"embedded\"");
        assert!(
            parse(&embedded)
                .unwrap_err()
                .downcast_ref::<LegacyEmbedded>()
                .is_some()
        );
        // Garbage stays an ordinary error.
        assert!(
            parse("stt = 3")
                .unwrap_err()
                .downcast_ref::<LegacyEmbedded>()
                .is_none()
        );
    }

    #[test]
    fn migration_keeps_connection_audio_and_hotkey() {
        let defaults = Choices::defaults_for(crate::setup::choices::Tier::Thin, "h").to_config();
        let cfg = migrate_legacy(LEGACY, &defaults).expect("migrates");
        assert_eq!(cfg.connection.ssh_destination, "me@box");
        assert_eq!(cfg.audio.input_device.as_deref(), Some("USB Mic"));
        assert!(cfg.hotkey.enabled && cfg.hotkey.mute == "Ctrl+Shift+M");
        assert_eq!(cfg.stt, defaults.stt);
        assert_eq!(cfg.tts, defaults.tts);
        assert!(
            migrate_legacy("[stt]\nkind = \"nemotron\"\nurl = \"ws://x\"\n", &defaults).is_none()
        );
    }

    #[test]
    fn null_in_extra_is_rejected() {
        let mut cfg = Choices::defaults_for(crate::setup::choices::Tier::Cpu, "h").to_config();
        let voicelink_backends::LlmConfig::OpenAi { extra, .. } = &mut cfg.llm;
        extra.insert("x".into(), serde_json::json!({"a": null}));
        assert!(render(&cfg).is_err());
    }

    #[test]
    fn save_is_atomic_and_private() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/config.toml");
        let cfg = Choices::defaults_for(crate::setup::choices::Tier::Cpu, "h").to_config();
        save(&path, &cfg).unwrap();
        assert_eq!(load(&path).unwrap(), cfg);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(leftovers.len(), 1, "{leftovers:?}");
    }
}
