//! Streaming and one-shot sample-rate conversion (rubato sinc, mono).

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, WindowFunction};

/// Input frames consumed per resampler call.
const CHUNK: usize = 480;

/// Converts a continuous mono stream from one rate to another.
///
/// Input may arrive in slices of any length; output is appended as soon as a
/// full internal chunk has been converted. A pass-through when the rates match.
pub(crate) struct StreamResampler {
    inner: Option<Async<f32>>,
    pending: Vec<f32>,
    out_buf: Vec<f32>,
}

impl StreamResampler {
    /// Creates a converter from `from` Hz to `to` Hz.
    pub(crate) fn new(from: u32, to: u32) -> anyhow::Result<Self> {
        anyhow::ensure!(from > 0 && to > 0, "sample rates must be non-zero");
        let inner = if from == to {
            None
        } else {
            let params = SincInterpolationParameters::new(128, WindowFunction::BlackmanHarris2);
            Some(Async::<f32>::new_sinc(
                f64::from(to) / f64::from(from),
                1.1,
                &params,
                CHUNK,
                1,
                FixedAsync::Input,
            )?)
        };
        let out_buf = inner
            .as_ref()
            .map(|r| vec![0.0; r.output_frames_max()])
            .unwrap_or_default();
        Ok(StreamResampler {
            inner,
            pending: Vec::new(),
            out_buf,
        })
    }

    /// Pushes `input` and appends every completed output sample to `out`.
    pub(crate) fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let Some(r) = self.inner.as_mut() else {
            out.extend_from_slice(input);
            return;
        };
        self.pending.extend_from_slice(input);
        loop {
            let need = r.input_frames_next();
            if self.pending.len() < need {
                break;
            }
            let Ok(ia) = InterleavedSlice::new(&self.pending[..need], 1, need) else {
                break;
            };
            let cap = self.out_buf.len();
            let Ok(mut oa) = InterleavedSlice::new_mut(&mut self.out_buf, 1, cap) else {
                break;
            };
            match r.process_into_buffer(&ia, &mut oa, None) {
                Ok((_, written)) => out.extend_from_slice(&self.out_buf[..written]),
                Err(e) => {
                    tracing::warn!("resampler error: {e}");
                    self.pending.clear();
                    return;
                }
            }
            self.pending.drain(..need);
        }
    }
}

impl StreamResampler {
    /// Pushes silence through to extract the buffered tail, then starts afresh.
    pub(crate) fn flush(&mut self, out: &mut Vec<f32>) {
        if self.inner.is_none() {
            return;
        }
        self.process(&vec![0.0; CHUNK * 2], out);
        self.pending.clear();
        if let Some(r) = self.inner.as_mut() {
            r.reset();
        }
    }
}

/// Resamples a whole mono clip, compensating the filter delay so the output
/// is time-aligned with the input and has `round(len * to / from)` samples.
pub(crate) fn resample_clip(samples: &[f32], from: u32, to: u32) -> anyhow::Result<Vec<f32>> {
    if from == to || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let mut rs = StreamResampler::new(from, to)?;
    let delay = rs.inner.as_ref().map(|r| r.output_delay()).unwrap_or(0);
    let want = ((samples.len() as f64) * f64::from(to) / f64::from(from)).round() as usize;
    let mut out = Vec::with_capacity(want + delay + CHUNK);
    rs.process(samples, &mut out);
    // Push silence until the delayed tail has been produced.
    let silence = vec![0.0f32; CHUNK];
    let mut guard = 0;
    while out.len() < want + delay && guard < 64 {
        rs.process(&silence, &mut out);
        guard += 1;
    }
    let start = delay.min(out.len());
    let mut result = out.split_off(start);
    result.truncate(want);
    result.resize(want, 0.0);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin() * 0.5)
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    #[test]
    fn downsample_keeps_tone_energy_and_length() {
        let input = tone(48_000, 440.0, 1.0);
        let out = resample_clip(&input, 48_000, 16_000).unwrap();
        assert_eq!(out.len(), 16_000);
        let r = rms(&out[2000..14_000]);
        assert!((r - 0.3535).abs() < 0.02, "rms {r}");
    }

    #[test]
    fn upsample_length_and_energy() {
        let input = tone(24_000, 300.0, 1.0);
        let out = resample_clip(&input, 24_000, 48_000).unwrap();
        assert_eq!(out.len(), 48_000);
        assert!((rms(&out[4000..44_000]) - 0.3535).abs() < 0.02);
    }

    #[test]
    fn out_of_band_tone_is_rejected_when_downsampling() {
        // 12 kHz is above the 8 kHz Nyquist of the target and must be filtered.
        let input = tone(48_000, 12_000.0, 1.0);
        let out = resample_clip(&input, 48_000, 16_000).unwrap();
        assert!(rms(&out[2000..14_000]) < 0.02);
    }

    #[test]
    fn passthrough_when_rates_match() {
        let input = tone(16_000, 200.0, 0.1);
        assert_eq!(resample_clip(&input, 16_000, 16_000).unwrap(), input);
    }

    #[test]
    fn stream_accepts_odd_slice_sizes() {
        let input = tone(44_100, 500.0, 1.0);
        let mut rs = StreamResampler::new(44_100, 16_000).unwrap();
        let mut out = Vec::new();
        for chunk in input.chunks(333) {
            rs.process(chunk, &mut out);
        }
        // Everything but the last partial chunk (and filter delay) is out.
        assert!(out.len() > 15_000 && out.len() <= 16_000, "{}", out.len());
    }
}
