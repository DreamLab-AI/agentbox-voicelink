//! Hardware probe and tier proposal.
//!
//! Probing shells out to `rocminfo`, `rocm-smi`, `nvidia-smi`, `vulkaninfo`
//! and `sysctl`; every parser is a pure function over captured text so the
//! tier logic is tested from fixtures.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::choices::Tier;

/// What the machine offers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HardwareReport {
    /// `linux` or `macos` (or other `std::env::consts::OS`).
    pub os: String,
    /// CPU architecture.
    pub arch: String,
    /// Logical CPUs.
    pub cpus: usize,
    /// System RAM in MiB.
    pub ram_mib: u64,
    /// ROCm GPU architecture names (`gfx1151`), if `rocminfo` ran.
    pub rocm_gfx: Vec<String>,
    /// Dedicated VRAM reported by `rocm-smi`, MiB.
    pub rocm_vram_mib: Option<u64>,
    /// GTT (system memory addressable by the GPU) reported by `rocm-smi`, MiB.
    pub rocm_gtt_mib: Option<u64>,
    /// NVIDIA GPUs as `(name, VRAM MiB)`.
    pub nvidia: Vec<(String, u64)>,
    /// Vulkan physical devices as `(name, type)`.
    pub vulkan: Vec<(String, String)>,
    /// Apple silicon with Metal.
    pub metal: bool,
}

/// A tier proposal with the evidence behind it.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    /// Proposed tier.
    pub tier: Tier,
    /// One line per observation.
    pub evidence: Vec<String>,
}

/// Runs `program args`, returning stdout when it exits successfully within `timeout`.
pub fn capture(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut out = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = out.read_to_string(&mut s);
        s
    });
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let text = reader.join().ok()?;
                return status.success().then_some(text);
            }
            Ok(None) if start.elapsed() < timeout => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

/// Probes the running machine.
pub fn probe() -> HardwareReport {
    let t = Duration::from_secs(4);
    let mut r = HardwareReport {
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        ..Default::default()
    };
    r.ram_mib = if cfg!(target_os = "macos") {
        capture("sysctl", &["-n", "hw.memsize"], t)
            .and_then(|s| parse_sysctl_memsize(&s))
            .unwrap_or(0)
    } else {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| parse_meminfo(&s))
            .unwrap_or(0)
    };
    r.metal = cfg!(target_os = "macos") && matches!(r.arch.as_str(), "aarch64" | "arm64");
    if let Some(s) = capture("rocminfo", &[], t) {
        r.rocm_gfx = parse_rocminfo(&s);
    }
    if let Some(s) = capture("rocm-smi", &["--showmeminfo", "vram", "gtt", "--json"], t) {
        let (v, g) = parse_rocm_smi_mem(&s);
        r.rocm_vram_mib = v;
        r.rocm_gtt_mib = g;
    }
    if let Some(s) = capture(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        t,
    ) {
        r.nvidia = parse_nvidia_smi(&s);
    }
    if let Some(s) = capture("vulkaninfo", &["--summary"], t) {
        r.vulkan = parse_vulkaninfo(&s);
    }
    r
}

/// `MemTotal` from `/proc/meminfo`, in MiB.
pub fn parse_meminfo(text: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

/// `sysctl -n hw.memsize` (bytes) to MiB.
pub fn parse_sysctl_memsize(text: &str) -> Option<u64> {
    Some(text.trim().parse::<u64>().ok()? / (1024 * 1024))
}

/// Distinct `gfxNNNN` agent names from `rocminfo`.
pub fn parse_rocminfo(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Name:") {
            let name = rest.trim();
            if name.starts_with("gfx") && !out.iter().any(|n| n == name) {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// `(vram, gtt)` totals in MiB from `rocm-smi --showmeminfo vram gtt --json`.
pub fn parse_rocm_smi_mem(text: &str) -> (Option<u64>, Option<u64>) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else {
        return (None, None);
    };
    let mut vram = None;
    let mut gtt = None;
    if let Some(cards) = v.as_object() {
        for card in cards.values() {
            let Some(o) = card.as_object() else { continue };
            for (k, val) in o {
                let bytes = val
                    .as_str()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .or_else(|| val.as_u64());
                let Some(b) = bytes else { continue };
                let mib = b / (1024 * 1024);
                if k.starts_with("VRAM Total Memory") {
                    vram = vram.max(Some(mib));
                } else if k.starts_with("GTT Total Memory") {
                    gtt = gtt.max(Some(mib));
                }
            }
        }
    }
    (vram, gtt)
}

/// `(name, MiB)` per line of the csv query.
pub fn parse_nvidia_smi(text: &str) -> Vec<(String, u64)> {
    text.lines()
        .filter_map(|l| {
            let (name, mem) = l.rsplit_once(',')?;
            Some((name.trim().to_string(), mem.trim().parse().ok()?))
        })
        .collect()
}

/// `(deviceName, deviceType)` pairs from `vulkaninfo --summary` (either field
/// order within a `GPUn:` block); software rasterisers (`llvmpipe`, type CPU)
/// are kept so the report can show them.
pub fn parse_vulkaninfo(text: &str) -> Vec<(String, String)> {
    fn flush(out: &mut Vec<(String, String)>, name: &mut Option<String>, ty: &mut Option<String>) {
        if let (Some(n), Some(t)) = (name.take(), ty.take()) {
            out.push((n, t));
        }
        *name = None;
        *ty = None;
    }
    let value = |v: &str| v.trim_start_matches([' ', '=']).trim().to_string();
    let mut out = Vec::new();
    let (mut name, mut ty) = (None, None);
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("GPU") && line.ends_with(':') {
            flush(&mut out, &mut name, &mut ty);
        } else if let Some(v) = line.strip_prefix("deviceName") {
            name = Some(value(v));
        } else if let Some(v) = line.strip_prefix("deviceType") {
            ty = Some(
                value(v)
                    .trim_start_matches("PHYSICAL_DEVICE_TYPE_")
                    .to_string(),
            );
        }
    }
    flush(&mut out, &mut name, &mut ty);
    out
}

impl HardwareReport {
    /// Memory the ROCm GPU can use, MiB. On an APU the dedicated VRAM is a
    /// small carve-out of the unified memory, so the whole of RAM counts.
    pub fn rocm_budget_mib(&self) -> Option<u64> {
        if self.rocm_gfx.is_empty() {
            return None;
        }
        let vram = self.rocm_vram_mib.unwrap_or(0);
        let unified = self.ram_mib > 0 && vram < self.ram_mib / 4;
        let gtt = self.rocm_gtt_mib.unwrap_or(0);
        Some(if unified {
            self.ram_mib.max(gtt)
        } else {
            vram.max(gtt)
        })
    }

    fn real_vulkan(&self) -> bool {
        self.vulkan
            .iter()
            .any(|(_, t)| !t.eq_ignore_ascii_case("CPU"))
    }

    /// Proposes a tier and explains why.
    pub fn propose(&self) -> Proposal {
        const GIB: u64 = 1024;
        // "32 GB" and "12 GB" are nominal: a 32 GB machine exposes about 31 GiB
        // and a 12 GB card reports 12 282 MiB, so the cut-offs sit slightly lower.
        const ROCM_LARGE_MIB: u64 = 30 * GIB;
        const NVIDIA_MIB: u64 = 11_500;
        let mut ev = vec![format!(
            "{} {}, {} logical CPUs, {:.1} GiB RAM",
            self.os,
            self.arch,
            self.cpus,
            self.ram_mib as f64 / 1024.0
        )];
        if !self.rocm_gfx.is_empty() {
            ev.push(format!(
                "ROCm: {} (vram {} MiB, gtt {} MiB, usable budget {} MiB)",
                self.rocm_gfx.join(", "),
                self.rocm_vram_mib.map_or("?".into(), |v| v.to_string()),
                self.rocm_gtt_mib.map_or("?".into(), |v| v.to_string()),
                self.rocm_budget_mib().unwrap_or(0)
            ));
        } else {
            ev.push("ROCm: not found (rocminfo)".into());
        }
        for (n, m) in &self.nvidia {
            ev.push(format!("NVIDIA: {n}, {m} MiB"));
        }
        if self.nvidia.is_empty() {
            ev.push("NVIDIA: not found (nvidia-smi)".into());
        }
        for (n, t) in &self.vulkan {
            ev.push(format!("Vulkan: {n} ({t})"));
        }
        if self.vulkan.is_empty() {
            ev.push("Vulkan: not found (vulkaninfo)".into());
        }
        if self.metal {
            ev.push("Metal: Apple silicon, unified memory".into());
        }
        let nvidia_best = self.nvidia.iter().map(|(_, m)| *m).max().unwrap_or(0);
        let (tier, why) = if self.ram_mib > 0 && self.ram_mib < 4 * GIB {
            (Tier::Thin, "under 4 GiB RAM: use remote speech")
        } else if self.rocm_budget_mib().unwrap_or(0) >= ROCM_LARGE_MIB {
            (
                Tier::RocmLarge,
                "ROCm GPU with at least 32 GiB unified or dedicated memory",
            )
        } else if nvidia_best >= NVIDIA_MIB {
            (Tier::Nvidia, "NVIDIA GPU with at least 12 GiB VRAM")
        } else if self.metal {
            (
                Tier::Metal,
                "Apple silicon: Metal-accelerated embedded speech",
            )
        } else if !self.rocm_gfx.is_empty() || !self.nvidia.is_empty() || self.real_vulkan() {
            (
                Tier::SmallGpu,
                "a GPU is present but below the large-model thresholds",
            )
        } else {
            (Tier::Cpu, "no usable GPU found")
        };
        ev.push(format!("Proposed tier: {} ({why})", tier.label()));
        Proposal { tier, evidence: ev }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> HardwareReport {
        HardwareReport {
            os: "linux".into(),
            arch: "x86_64".into(),
            cpus: 16,
            ram_mib: 64 * 1024,
            ..Default::default()
        }
    }

    #[test]
    fn strix_halo_is_rocm_large() {
        let r = HardwareReport {
            rocm_gfx: vec!["gfx1151".into()],
            rocm_vram_mib: Some(512),
            rocm_gtt_mib: Some(60 * 1024),
            vulkan: vec![("AMD Radeon 8060S".into(), "INTEGRATED_GPU".into())],
            ..base()
        };
        assert_eq!(r.propose().tier, Tier::RocmLarge);
        assert!(r.propose().evidence.iter().any(|l| l.contains("gfx1151")));
    }

    #[test]
    fn small_unified_memory_is_not_rocm_large() {
        let r = HardwareReport {
            rocm_gfx: vec!["gfx1103".into()],
            rocm_vram_mib: Some(512),
            ram_mib: 16 * 1024,
            ..base()
        };
        assert_eq!(r.propose().tier, Tier::SmallGpu);
    }

    #[test]
    fn discrete_rocm_uses_vram() {
        let r = HardwareReport {
            rocm_gfx: vec!["gfx1100".into()],
            rocm_vram_mib: Some(24 * 1024),
            rocm_gtt_mib: Some(0),
            ..base()
        };
        assert_eq!(r.rocm_budget_mib(), Some(24 * 1024));
        assert_eq!(r.propose().tier, Tier::SmallGpu);
    }

    #[test]
    fn nvidia_threshold() {
        let big = HardwareReport {
            nvidia: vec![("RTX 4070".into(), 12282)],
            ..base()
        };
        assert_eq!(big.propose().tier, Tier::Nvidia);
        let small = HardwareReport {
            nvidia: vec![("GTX 1650".into(), 4096)],
            ..base()
        };
        assert_eq!(small.propose().tier, Tier::SmallGpu);
    }

    #[test]
    fn apple_silicon_is_metal() {
        let r = HardwareReport {
            os: "macos".into(),
            arch: "aarch64".into(),
            metal: true,
            ram_mib: 16 * 1024,
            ..base()
        };
        assert_eq!(r.propose().tier, Tier::Metal);
    }

    #[test]
    fn software_vulkan_is_cpu_and_tiny_ram_is_thin() {
        let soft = HardwareReport {
            vulkan: vec![("llvmpipe".into(), "CPU".into())],
            ..base()
        };
        assert_eq!(soft.propose().tier, Tier::Cpu);
        let tiny = HardwareReport {
            ram_mib: 2048,
            ..base()
        };
        assert_eq!(tiny.propose().tier, Tier::Thin);
        assert_eq!(
            HardwareReport {
                vulkan: vec![("Intel".into(), "INTEGRATED_GPU".into())],
                ..base()
            }
            .propose()
            .tier,
            Tier::SmallGpu
        );
    }

    #[test]
    fn parsers() {
        assert_eq!(
            parse_meminfo("MemTotal:       65536000 kB\nMemFree: 1 kB\n"),
            Some(64000)
        );
        assert_eq!(parse_sysctl_memsize("17179869184\n"), Some(16384));
        let rocminfo = "Agent 1\n  Name:                    AMD Ryzen\nAgent 2\n  Name:                    gfx1151\n  Marketing Name: x\nAgent 3\n  Name: gfx1151\n";
        assert_eq!(parse_rocminfo(rocminfo), vec!["gfx1151"]);
        let smi = r#"{"card0": {"VRAM Total Memory (B)": "536870912", "VRAM Total Used Memory (B)": "1", "GTT Total Memory (B)": "64424509440"}}"#;
        assert_eq!(parse_rocm_smi_mem(smi), (Some(512), Some(61440)));
        assert_eq!(
            parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564\n"),
            vec![("NVIDIA GeForce RTX 4090".to_string(), 24564)]
        );
        let vk = "GPU0:\n\tapiVersion = 1.3\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n\tdeviceName         = AMD Radeon 8060S\n";
        assert_eq!(
            parse_vulkaninfo(vk),
            vec![("AMD Radeon 8060S".to_string(), "INTEGRATED_GPU".to_string())]
        );
        let vk2 = "GPU0:\n\tdeviceName         = AMD Radeon 8060S\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n";
        assert_eq!(
            parse_vulkaninfo(vk2),
            vec![("AMD Radeon 8060S".to_string(), "INTEGRATED_GPU".to_string())]
        );
        assert_eq!(parse_meminfo("nothing"), None);
    }
}
