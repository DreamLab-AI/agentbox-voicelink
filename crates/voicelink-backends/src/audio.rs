//! Resampling and WAV helpers.

use crate::{AudioClip, BackendError, Result};
use rubato::{FftFixedIn, Resampler};

/// Resamples `clip` to `rate` Hz with a band-limited FFT resampler.
///
/// The output length is `round(len * rate / sample_rate)`. A clip already at
/// `rate` is returned unchanged (cloned).
///
/// # Examples
///
/// ```
/// use voicelink_backends::{resample, AudioClip};
///
/// let clip = AudioClip { sample_rate: 48_000, samples: vec![0.0; 4_800] };
/// let out = resample(&clip, 16_000);
/// assert_eq!(out.sample_rate, 16_000);
/// assert_eq!(out.samples.len(), 1_600);
/// ```
pub fn resample(clip: &AudioClip, rate: u32) -> AudioClip {
    if clip.sample_rate == rate {
        return clip.clone();
    }
    if clip.samples.is_empty() || clip.sample_rate == 0 || rate == 0 {
        return AudioClip {
            sample_rate: rate,
            samples: Vec::new(),
        };
    }
    let expected = ((clip.samples.len() as u64 * rate as u64 + clip.sample_rate as u64 / 2)
        / clip.sample_rate as u64) as usize;
    const CHUNK: usize = 1024;
    let Ok(mut rs) = FftFixedIn::<f32>::new(clip.sample_rate as usize, rate as usize, CHUNK, 2, 1)
    else {
        return linear(clip, rate, expected);
    };
    let delay = rs.output_delay();
    let mut out: Vec<f32> = Vec::with_capacity(expected + delay + CHUNK);
    let mut pos = 0;
    while clip.samples.len() - pos >= rs.input_frames_next() {
        let n = rs.input_frames_next();
        match rs.process(&[&clip.samples[pos..pos + n]], None) {
            Ok(b) => out.extend_from_slice(&b[0]),
            Err(_) => return linear(clip, rate, expected),
        }
        pos += n;
    }
    let rest: &[f32] = &clip.samples[pos..];
    let rest_buf = [rest];
    let input = if rest.is_empty() {
        None
    } else {
        Some(&rest_buf[..])
    };
    if let Ok(b) = rs.process_partial(input, None) {
        out.extend_from_slice(&b[0]);
    }
    // Flush the filter tail until the delayed signal is fully emitted.
    while out.len() < expected + delay {
        match rs.process_partial::<&[f32]>(None, None) {
            Ok(b) if !b[0].is_empty() => out.extend_from_slice(&b[0]),
            _ => break,
        }
    }
    let mut samples: Vec<f32> = out.into_iter().skip(delay).take(expected).collect();
    samples.resize(expected, 0.0);
    AudioClip {
        sample_rate: rate,
        samples,
    }
}

fn linear(clip: &AudioClip, rate: u32, expected: usize) -> AudioClip {
    let ratio = clip.sample_rate as f64 / rate as f64;
    let samples = (0..expected)
        .map(|i| {
            let p = i as f64 * ratio;
            let i0 = p.floor() as usize;
            let f = (p - i0 as f64) as f32;
            let a = clip.samples.get(i0).copied().unwrap_or(0.0);
            let b = clip.samples.get(i0 + 1).copied().unwrap_or(a);
            a + (b - a) * f
        })
        .collect();
    AudioClip {
        sample_rate: rate,
        samples,
    }
}

/// Converts one float sample to signed 16-bit PCM with clipping.
pub(crate) fn f32_to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * 32767.0).round() as i16
}

/// Encodes `clip` as a 16-bit mono PCM WAV file.
///
/// # Examples
///
/// ```
/// use voicelink_backends::{decode_wav, encode_wav, AudioClip};
///
/// let clip = AudioClip { sample_rate: 16_000, samples: vec![0.0, 0.5, -0.5] };
/// let wav = encode_wav(&clip);
/// let back = decode_wav(&wav).unwrap();
/// assert_eq!(back.sample_rate, 16_000);
/// assert_eq!(back.samples.len(), 3);
/// ```
pub fn encode_wav(clip: &AudioClip) -> Vec<u8> {
    let data_len = (clip.samples.len() * 2) as u32;
    let mut v = Vec::with_capacity(44 + data_len as usize);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes()); // PCM
    v.extend_from_slice(&1u16.to_le_bytes()); // mono
    v.extend_from_slice(&clip.sample_rate.to_le_bytes());
    v.extend_from_slice(&(clip.sample_rate * 2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for s in &clip.samples {
        v.extend_from_slice(&f32_to_i16(*s).to_le_bytes());
    }
    v
}

/// Decodes a PCM WAV file (8/16/24/32-bit integer or 32-bit float, any channel
/// count) into a mono [`AudioClip`], averaging channels.
///
/// # Errors
///
/// [`BackendError::Backend`] when the data is not a supported WAV file.
pub fn decode_wav(bytes: &[u8]) -> Result<AudioClip> {
    let bad = |m: &str| BackendError::Backend(format!("wav: {m}"));
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let u32_at =
        |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    let mut pos = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let len = u32_at(pos + 4) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            let mut tag = u16_at(body);
            if tag == 0xFFFE && len >= 26 && body + 26 <= bytes.len() {
                tag = u16_at(body + 24);
            }
            fmt = Some((tag, u16_at(body + 2), u32_at(body + 4), u16_at(body + 14)));
        } else if id == b"data" {
            let (tag, ch, rate, bits) = fmt.ok_or_else(|| bad("data before fmt"))?;
            let end = body.saturating_add(len).min(bytes.len());
            let data = &bytes[body..end];
            let ch = ch.max(1) as usize;
            let width = (bits / 8) as usize;
            if width == 0 || !matches!((tag, bits), (1, 8 | 16 | 24 | 32) | (3, 32)) {
                return Err(bad("unsupported sample format"));
            }
            let frames = data.len() / (width * ch);
            let mut samples = Vec::with_capacity(frames);
            for f in 0..frames {
                let mut acc = 0.0f32;
                for c in 0..ch {
                    let o = (f * ch + c) * width;
                    let b = &data[o..o + width];
                    acc += match (tag, bits) {
                        (1, 8) => (b[0] as f32 - 128.0) / 128.0,
                        (1, 16) => i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
                        (1, 24) => {
                            (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0
                        }
                        (1, _) => {
                            i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
                        }
                        _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                    };
                }
                samples.push(acc / ch as f32);
            }
            return Ok(AudioClip {
                sample_rate: rate,
                samples,
            });
        }
        pos = body.saturating_add(len + (len & 1));
    }
    Err(bad("no data chunk"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, secs: f32, hz: f32) -> AudioClip {
        let n = (rate as f32 * secs) as usize;
        AudioClip {
            sample_rate: rate,
            samples: (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin() * 0.5)
                .collect(),
        }
    }

    #[test]
    fn resample_lengths_and_identity() {
        for (from, to) in [
            (48_000, 16_000),
            (24_000, 16_000),
            (44_100, 16_000),
            (16_000, 24_000),
        ] {
            let c = sine(from, 0.5, 440.0);
            let r = resample(&c, to);
            assert_eq!(r.sample_rate, to);
            assert_eq!(
                r.samples.len(),
                (c.samples.len() as u64 * to as u64 / from as u64) as usize
            );
        }
        let c = sine(16_000, 0.1, 440.0);
        assert_eq!(resample(&c, 16_000), c);
        assert!(
            resample(
                &AudioClip {
                    sample_rate: 8000,
                    samples: vec![]
                },
                16_000
            )
            .samples
            .is_empty()
        );
    }

    #[test]
    fn resample_preserves_tone_energy() {
        let c = sine(48_000, 0.5, 440.0);
        let r = resample(&c, 16_000);
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let (a, b) = (rms(&c.samples), rms(&r.samples[400..r.samples.len() - 400]));
        assert!((a - b).abs() < 0.02, "{a} vs {b}");
    }

    #[test]
    fn wav_roundtrip() {
        let c = sine(16_000, 0.1, 300.0);
        let d = decode_wav(&encode_wav(&c)).unwrap();
        assert_eq!(d.sample_rate, 16_000);
        assert_eq!(d.samples.len(), c.samples.len());
        for (x, y) in c.samples.iter().zip(&d.samples) {
            assert!((x - y).abs() < 1e-3);
        }
        assert!(decode_wav(b"garbage").is_err());
    }
}
