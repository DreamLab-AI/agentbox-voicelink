//! In-process acoustic echo cancellation (pure-Rust WebRTC AEC3 port, `aec3`).
//!
//! Runs at 16 kHz mono on 10 ms frames. The far-end reference is the audio the
//! playback path actually emitted, so queue contents that were cleared by
//! barge-in never count as echo.

use aec3::nodes::audio::AudioFormat;
use aec3::pipelines::linear::{self, LinearPipeline};

/// Sample rate the canceller works at.
pub(crate) const AEC_RATE: u32 = 16_000;
/// Samples in one 10 ms frame at [`AEC_RATE`].
pub(crate) const AEC_FRAME: usize = 160;

/// Echo canceller for one mono capture stream.
pub(crate) struct EchoCanceller {
    pipeline: LinearPipeline,
}

impl EchoCanceller {
    /// Builds a canceller. `initial_delay_ms` is the expected output-to-input
    /// path delay; the AEC refines it on its own.
    pub(crate) fn new(initial_delay_ms: i32) -> anyhow::Result<Self> {
        let format = AudioFormat::ten_ms(AEC_RATE, 1);
        let pipeline = linear::builder(format, format)
            .initial_delay_ms(initial_delay_ms)
            // Keep the signal level honest for the VAD: no gain control or
            // noise suppression, only the high-pass filter and the AEC.
            .enable_gain_controller2(false)
            .enable_noise_suppression(false)
            .build()
            .map_err(|e| anyhow::anyhow!("echo canceller: {e}"))?;
        Ok(EchoCanceller { pipeline })
    }

    /// Cleans one 10 ms capture frame given the matching far-end frame.
    ///
    /// Both slices must hold exactly [`AEC_FRAME`] samples. On an internal
    /// error the capture frame is returned unchanged.
    pub(crate) fn process(&mut self, far: &[f32], near: &[f32]) -> Vec<f32> {
        debug_assert_eq!(far.len(), AEC_FRAME);
        debug_assert_eq!(near.len(), AEC_FRAME);
        if let Err(e) = self.pipeline.handle_render_frame(far) {
            tracing::debug!("aec render: {e}");
            return near.to_vec();
        }
        let mut out = vec![0.0f32; AEC_FRAME];
        match self.pipeline.process_capture_frame(near, &mut out) {
            Ok(true) => out,
            Ok(false) => near.to_vec(),
            Err(e) => {
                tracing::debug!("aec capture: {e}");
                near.to_vec()
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deterministic white noise in [-amp, amp].
    pub(crate) fn noise(len: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as f32) / (u32::MAX >> 1) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    fn run(far: &[f32], near: &[f32], delay_ms: i32) -> Vec<f32> {
        let mut aec = EchoCanceller::new(delay_ms).unwrap();
        let mut out = Vec::new();
        let (far, _) = far.as_chunks::<AEC_FRAME>();
        let (near, _) = near.as_chunks::<AEC_FRAME>();
        for (f, n) in far.iter().zip(near) {
            out.extend(aec.process(f, n));
        }
        out
    }

    #[test]
    fn echo_of_played_signal_is_attenuated() {
        let secs = 8;
        let n = AEC_RATE as usize * secs;
        let delay = 800; // 50 ms
        let far = noise(n, 0.3, 7);
        let mut near = vec![0.0f32; n];
        for i in delay..n {
            near[i] = far[i - delay] * 0.5;
        }
        let out = run(&far, &near, 50);
        let tail = n - AEC_RATE as usize * 2..n;
        let before = rms(&near[tail.clone()]);
        let after = rms(&out[tail]);
        assert!(
            after < before * 0.35,
            "echo not attenuated: {before} -> {after}"
        );
    }

    #[test]
    fn near_end_speech_survives_when_nothing_is_played() {
        let n = AEC_RATE as usize * 2;
        let far = vec![0.0f32; n];
        let near = noise(n, 0.2, 3);
        let out = run(&far, &near, 0);
        let tail = n / 2..n;
        assert!(rms(&out[tail.clone()]) > rms(&near[tail]) * 0.5);
    }
}
