//! `voicelink devices`: the audio devices voicelink can use, by the names
//! `[audio] input_device` and `output_device` accept.

use crate::audio::{self, DeviceList};

/// Prints the input and output devices, marking the system defaults.
pub fn run() -> anyhow::Result<()> {
    print!("{}", render(&audio::list_devices()?));
    Ok(())
}

fn section(out: &mut String, title: &str, names: &[String], default: Option<&str>) {
    out.push_str(title);
    out.push('\n');
    if names.is_empty() {
        out.push_str("  (none)\n");
    }
    for name in names {
        let mark = if Some(name.as_str()) == default {
            "  (system default)"
        } else {
            ""
        };
        out.push_str(&format!("  {name}{mark}\n"));
    }
}

fn render(list: &DeviceList) -> String {
    let mut out = String::new();
    section(
        &mut out,
        "Input devices:",
        &list.inputs,
        list.default_input.as_deref(),
    );
    section(
        &mut out,
        "Output devices:",
        &list.outputs,
        list.default_output.as_deref(),
    );
    out.push_str(
        "\nTo use one, set input_device or output_device under [audio] in the config\n\
         (or run `voicelink setup`), then reconnect. Leave them unset to follow the\n\
         system default.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_the_defaults() {
        let list = DeviceList {
            inputs: vec!["USB Mic".into(), "Built-in".into()],
            outputs: vec![],
            default_input: Some("USB Mic".into()),
            default_output: None,
        };
        let text = render(&list);
        assert!(text.contains("  USB Mic  (system default)\n"), "{text}");
        assert!(text.contains("  Built-in\n"), "{text}");
        assert!(text.contains("Output devices:\n  (none)\n"), "{text}");
    }
}
