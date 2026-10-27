//! The downloadable model catalogue: whisper.cpp ggml models for the compose
//! stack's `whisper-server`.
//!
//! Every entry carries the exact URL and the SHA-256 of the file, the Git LFS
//! `oid` values reported by the Hugging Face API
//! (`GET /api/models/ggerganov/whisper.cpp/tree/main`). There are no Kokoro
//! entries: the Kokoro-FastAPI images bundle their weights. There are no LLM
//! entries either: the GGUF is the operator's choice, placed in the models
//! directory (`compose/README.md`).

/// One downloadable file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// Stable identifier used in flags and wizard state.
    pub id: &'static str,
    /// Human label.
    pub label: &'static str,
    /// File name inside the models directory.
    pub file: &'static str,
    /// `https` source URL.
    pub url: &'static str,
    /// Lower-case hex SHA-256 of the whole file.
    pub sha256: &'static str,
    /// Size in bytes.
    pub size: u64,
    /// Licence of the weights.
    pub licence: &'static str,
}

/// Every model the wizard can fetch, in menu order.
pub const CATALOGUE: &[Entry] = &[
    Entry {
        id: "whisper-large-v3-turbo",
        label: "Whisper large-v3-turbo (f16, 1.6 GB)",
        file: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
        sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        size: 1_624_555_275,
        licence: "MIT",
    },
    Entry {
        id: "whisper-large-v3-turbo-q8",
        label: "Whisper large-v3-turbo (q8_0, 874 MB)",
        file: "ggml-large-v3-turbo-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin",
        sha256: "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
        size: 874_188_075,
        licence: "MIT",
    },
    Entry {
        id: "whisper-large-v3-turbo-q5",
        label: "Whisper large-v3-turbo (q5_0, 574 MB)",
        file: "ggml-large-v3-turbo-q5_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        sha256: "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        size: 574_041_195,
        licence: "MIT",
    },
    Entry {
        id: "whisper-medium-q8",
        label: "Whisper medium (q8_0, 823 MB)",
        file: "ggml-medium-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q8_0.bin",
        sha256: "42a1ffcbe4167d224232443396968db4d02d4e8e87e213d3ee2e03095dea6502",
        size: 823_369_779,
        licence: "MIT",
    },
    Entry {
        id: "whisper-small-q8",
        label: "Whisper small (q8_0, 264 MB)",
        file: "ggml-small-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q8_0.bin",
        sha256: "49c8fb02b65e6049d5fa6c04f81f53b867b5ec9540406812c643f177317f779f",
        size: 264_464_607,
        licence: "MIT",
    },
    Entry {
        id: "whisper-base-en-q8",
        label: "Whisper base.en (q8_0, 82 MB, English only)",
        file: "ggml-base.en-q8_0.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q8_0.bin",
        sha256: "a4d4a0768075e13cfd7e19df3ae2dbc4a68d37d36a7dad45e8410c9a34f8c87e",
        size: 81_781_811,
        licence: "MIT",
    },
];

/// Looks an entry up by id.
pub fn find(id: &str) -> Option<&'static Entry> {
    CATALOGUE.iter().find(|e| e.id == id)
}

/// Looks an entry up by file name.
pub fn find_by_file(file: &str) -> Option<&'static Entry> {
    CATALOGUE.iter().find(|e| e.file == file)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREFIX: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

    #[test]
    fn every_entry_is_well_formed() {
        for e in CATALOGUE {
            assert_eq!(e.sha256.len(), 64, "{}", e.id);
            assert!(
                e.sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{}",
                e.id
            );
            assert!(e.url.starts_with("https://"), "{}", e.id);
            assert!(e.url.starts_with(PREFIX), "{}", e.id);
            assert!(e.url.ends_with(e.file), "{}", e.id);
            assert!(!e.file.contains('/'), "{}", e.id);
            assert!(e.size > 1_000_000, "{}", e.id);
        }
    }

    #[test]
    fn ids_and_files_are_unique() {
        for (i, a) in CATALOGUE.iter().enumerate() {
            for b in &CATALOGUE[i + 1..] {
                assert_ne!(a.id, b.id);
                assert_ne!(a.file, b.file);
                assert_ne!(a.sha256, b.sha256);
            }
        }
    }

    #[test]
    fn lookups() {
        assert_eq!(
            find("whisper-small-q8").unwrap().file,
            "ggml-small-q8_0.bin"
        );
        assert!(find_by_file("ggml-small-q8_0.bin").is_some());
        assert!(find("kokoro-model").is_none());
        assert!(CATALOGUE.len() >= 4);
        // The compose default must stay downloadable.
        assert!(find_by_file(crate::stack::DEFAULT_WHISPER_FILE).is_some());
    }
}
