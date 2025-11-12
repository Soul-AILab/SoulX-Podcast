# Changelog

All notable changes to this project are documented in this file.
This project follows a simplified changelog approach. New, unreleased
changes are listed under the `Unreleased` heading.

## Unreleased

### Added
- Sentence-level streaming CLI with REPL and live playback: `cli/streaming.py`.
  - Flags: `--text`, `--repl`, `--play`, `--reference_wav`, `--precache_prompts`, `--clear_cache`, `--timings`.
- Non-blocking REPL input queue: users can type next prompts while the model is busy.
- Producer-consumer pipeline that decouples model compute and audio playback to reduce GPU idle gaps.
- Prompt-feature persistent cache (stored under `.prompt_cache/`) using WAV basename filenames; cache entries include `mtime` metadata for validation.
- Lightweight timing collector and `--timings` flag to report per-stage timings.
- Friendly system status messages for model loading, precache progress, and REPL readiness.
- README updated with streaming CLI documentation and usage examples.

### Changed
- Improved CLI UX for non-developers: clear status messages and queued input confirmations.

### Notes
- Persisted prompt cache validates by file modification time (mtime). If you prefer exact binary matching, consider SHA1-based validation (can be added as an option).
- Optional playback dependencies (`sounddevice` or `simpleaudio`) are recommended for live playback.


## 2025-10-31
- Deploy an online demo on Hugging Face Spaces.


(End of changelog)
