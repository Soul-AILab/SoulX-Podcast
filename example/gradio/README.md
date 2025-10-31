Gradio UI for SoulX-Podcast
==========================

Quick instructions to run the Gradio UI locally (recommended to run inside the project's virtualenv in WSL/Ubuntu where dependencies are installed):

1. Ensure dependencies are installed (project `requirements.txt` already includes `gradio`).

   Gradio UI for SoulX-Podcast
   ==========================

   This folder contains a small demo of the Gradio UI and quick usage notes.

   Formatting the dialogue textarea

   - For Dual Speaker mode you can enter multi-line dialogue where each line is either prefixed with the speaker label (S1: / S2:) or uses the bracketed tag format `[S1]` / `[S2]`. Examples:

      S1: Hello, welcome to the show.
      S2: Thanks — happy to be here.

      or

      [S1] Hello, welcome to the show.
      [S2] Thanks — happy to be here.

   - The UI will convert either form into the canonical internal speaker-tagged form the pipeline expects (e.g. `[S1]Hello, welcome to the show.`).

   Speaker mapping

   - S1 -> Speaker 1 (first reference audio / text)
   - S2 -> Speaker 2 (second reference audio / text)

   Uploading two reference audios and texts

   - Dual Speaker tab allows you to upload two reference audio files (one per speaker) and optional reference text for each speaker.
   - If you only upload one reference audio, the other speaker will use a silent fallback. To get best results, provide short (1-10s) clean reference audio for each speaker.

   Expected behavior

   - Single Speaker tab:
      - Use `Manual utterance` (or upload JSON) and optionally supply one reference audio / text.
      - The app will synthesize a single utterance and save the output to `outputs/output_single.wav` by default.

   - Dual Speaker tab:
      - Enter a multi-line dialogue using `S1:` and `S2:` prefixes or upload a small JSON.
      - Provide 0, 1, or 2 reference audios and optional texts. The app will attempt to use speaker-specific references in order.
      - Output is saved to `outputs/output_dual.wav` by default.

   Smoke test (quick, non-heavy check)

   A lightweight smoke-test script is provided to validate UI input handling and fallback audio creation without running the full model by default. To run the light smoke test (does not load model):

   ```powershell
   python example/gradio/smoke_test.py
   ```

   To run the full inference (this can be heavy and will load the selected model), add `--run` and optionally set `--model-path`:

   ```powershell
   python example/gradio/smoke_test.py --run --model-path pretrained_models/SoulX-Podcast-1.7B
   ```

   Notes

   - Dual-speaker synthesis will be more reliable when both speakers have clean reference audio.
   - The UI keeps single-speaker behavior unchanged; the Dual Speaker tab is optional and additive.

   Quick start
   -----------

   1. Install dependencies in your venv (WSL recommended):

   ```powershell
   pip install -r requirements.txt
   ```

   2. Run the Gradio app from the project root:

   ```bash
   python gradio_app.py
   ```

   3. Open http://localhost:7860 in your browser.

   Example JSON
   ------------

   An example JSON for the Dual Speaker tab (mixed setup: Speaker 1 uses a preloaded sample, Speaker 2 is an uploaded file path) is provided at `example/gradio/dialog_example_mixed.json`. Replace the `prompt_audio` path for `S2` with your uploaded file path or upload via the UI.

   ```json
   {
      "speakers": {
         "S1": { "prompt_text": "A calm female host", "prompt_audio": "example/audios/female_mandarin.wav" },
         "S2": { "prompt_text": "A guest voice", "prompt_audio": "/path/to/your/uploaded_speaker2.wav" }
      },
      "text": [["S1","Hello..."],["S2","Hi..."]]
   }
   ```

   Referencing repo samples from JSON
   ----------------------------------

   You can reference the preloaded audio samples (those listed in `example/audios`) using short tokens in your JSON. The Gradio UI will resolve the tokens to the currently selected UI samples when you run the job.

   - Use `"sample1"` to refer to Speaker 1's selected sample.
   - Use `"sample2"` to refer to Speaker 2's selected sample.

   Example JSON using token references (upload this file in the Dual Speaker tab):

   ```json
   {
      "speakers": {
         "S1": { "prompt_text": "Host voice (uses sample1 token)", "prompt_audio": "sample1" },
         "S2": { "prompt_text": "Guest voice (uses sample2 token)", "prompt_audio": "sample2" }
      },
      "text": [["S1","Welcome to the show."],["S2","Thanks for having me."]]
   }
   ```

   There is also a ready example file you can upload: `dialog_example_sample_tokens.json` which uses `sample1` and `sample2` tokens. Make sure to pick the desired samples in the Dual Speaker UI dropdowns before running the job so the tokens resolve to the audio files you want.
