# Changelog

All notable changes to this repository should be documented in this file.

## [Unreleased] - feature/gradio-ui (2025-10-30)

  - Accepts either a structured podcast JSON (recommended) or a manual utterance.
  - Supports optional reference audio/text and dialect prompts.
  - Improved, more descriptive labels and user guidance in the UI.

Notes for upstream maintainers:

### Added (2025-10-31)

- Document non-verbal/paralinguistic token support (e.g., <|laughter|>, <|breathing|>, <|coughing|>) in the README and examples. These tokens can be inserted into input text to control paralinguistic events during generation.
 - Added Gradio UI Single / Dual Speaker tabs. Dual Speaker supports uploading two reference audios/texts and a dialogue textarea where lines can be prefixed with `S1:` and `S2:` to indicate speaker turns.
 - Added `example/gradio/README.md` with usage examples and guidance for formatting dialogue and reference audio usage.
 - Added `example/gradio/smoke_test.py` — lightweight smoke test to validate UI input handling and optionally run a full inference when `--run` is supplied.
 - Updated README to document Gradio UI usage and bracketed `[S1]/[S2]` dialog tag format; linked to `example/gradio/README.md` and smoke test.
