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

/// Parses config text.
pub fn parse(text: &str) -> anyhow::Result<Config> {
    Ok(toml::from_str(text)?)
}

/// Serialises a config. TOML has no null, so JSON nulls in the LLM `extra`
/// map are rejected rather than silently dropped.
pub fn render(config: &Config) -> anyhow::Result<String> {
    #[allow(irrefutable_let_patterns)]
    if let voicelink_backends::LlmConfig::OpenAi { extra, .. } = &config.llm
        && extra.values().any(contains_null)
    {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::choices::Choices;

    #[test]
    fn round_trips_every_variant() {
        let mut c = Choices::defaults_for(crate::setup::choices::Tier::Cpu, "host");
        for stt in [
            crate::setup::choices::SttKind::Embedded,
            crate::setup::choices::SttKind::Http,
            crate::setup::choices::SttKind::Nemotron,
        ] {
            for tts in [
                crate::setup::choices::TtsKind::Embedded,
                crate::setup::choices::TtsKind::Http,
                crate::setup::choices::TtsKind::Pocket,
            ] {
                for llm in [
                    crate::setup::choices::LlmKind::Local,
                    crate::setup::choices::LlmKind::Compose,
                    crate::setup::choices::LlmKind::Loom,
                ] {
                    c.stt_kind = stt;
                    c.tts_kind = tts;
                    c.llm_kind = llm;
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
    fn null_in_extra_is_rejected() {
        let mut cfg = Choices::defaults_for(crate::setup::choices::Tier::Cpu, "h").to_config();
        #[allow(irrefutable_let_patterns)]
        let voicelink_backends::LlmConfig::OpenAi { extra, .. } = &mut cfg.llm else {
            unreachable!()
        };
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
