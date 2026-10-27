//! Embedded whisper.cpp speech-to-text (feature `whisper*`).

use crate::{AudioClip, BackendError, Result, SpeechToText, Transcript, resample};
use async_trait::async_trait;
use std::sync::Arc;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub(crate) struct WhisperStt {
    ctx: Arc<WhisperContext>,
    language: Option<String>,
    threads: i32,
    model: String,
}

impl WhisperStt {
    pub(crate) fn new(
        model_path: &str,
        language: &Option<String>,
        threads: u16,
        gpu: bool,
    ) -> Result<Self> {
        if !std::path::Path::new(model_path).is_file() {
            return Err(BackendError::Config(format!(
                "whisper model not found: {model_path}"
            )));
        }
        // whisper.cpp logs to stderr by default; a client must never write to the terminal.
        whisper_rs::install_logging_hooks();
        let mut p = WhisperContextParameters::default();
        p.use_gpu(gpu && cfg!(any(feature = "whisper-vulkan", feature = "whisper-metal")));
        let ctx = WhisperContext::new_with_params(model_path, p)
            .map_err(|e| BackendError::Config(format!("whisper model {model_path}: {e}")))?;
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(4, |n| n.get().min(8) as i32)
        } else {
            i32::from(threads)
        };
        Ok(Self {
            ctx: Arc::new(ctx),
            language: language.clone().filter(|l| !l.is_empty() && l != "auto"),
            threads,
            model: std::path::Path::new(model_path)
                .file_stem()
                .map_or_else(String::new, |s| s.to_string_lossy().into_owned()),
        })
    }
}

#[async_trait]
impl SpeechToText for WhisperStt {
    async fn transcribe(&self, clip: &AudioClip) -> Result<Transcript> {
        let mut samples = resample(clip, 16_000).samples;
        // whisper.cpp rejects input shorter than 1 s; pad with silence.
        if samples.len() < 16_800 {
            samples.resize(16_800, 0.0);
        }
        let ctx = Arc::clone(&self.ctx);
        let language = self.language.clone();
        let threads = self.threads;
        tokio::task::spawn_blocking(move || {
            let mut state = ctx
                .create_state()
                .map_err(|e| BackendError::Backend(format!("whisper state: {e}")))?;
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_n_threads(threads);
            params.set_language(language.as_deref());
            params.set_translate(false);
            params.set_no_context(true);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            state
                .full(params, &samples)
                .map_err(|e| BackendError::Backend(format!("whisper inference: {e}")))?;
            let mut text = String::new();
            for seg in state.as_iter() {
                text.push_str(
                    &seg.to_str_lossy()
                        .map_err(|e| BackendError::Backend(e.to_string()))?,
                );
            }
            Ok(Transcript {
                text: text.trim().to_owned(),
            })
        })
        .await
        .map_err(|e| BackendError::Backend(format!("whisper task: {e}")))?
    }

    fn name(&self) -> String {
        format!("whisper.cpp:{}", self.model)
    }
}
