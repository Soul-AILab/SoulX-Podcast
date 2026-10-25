# cwcge-pdf2audio

`cwcge-pdf2audio` is a PDF-to-speech application for producing long-form audio versions of lecture notes and academic reading material used by the General Education Units of CW Chu College.

The project was developed in response to requests from **Professor Kenneth Young**, under the guidance of **Professor Tan Lee** and **Paco Fung**. The application combines academic-document text extraction and cleaning with the SoulX-Podcast speech-generation model so that a user can prepare a born-digital academic PDF, review the text that will actually be spoken, and generate long-form speech using a supplied reference voice.

> This repository contains application code derived from and built on top of SoulX-Podcast. The original SoulX-Podcast model/inference code remains subject to the repository's Apache 2.0 license. See `LICENSE` and the acknowledgements below.

---

## 1. Purpose

Academic lecture notes and readings are not directly suitable for text-to-speech. A PDF may contain page numbers, repeated headers and footers, references, bibliography entries, footnotes, figure/table material, inline citations, line-break hyphenation, and other content that should not necessarily be read aloud.

This project therefore separates the task into two stages:

1. **PDF → speech-ready text**
   - extract readable text from an academic PDF;
   - remove or reduce non-spoken academic material;
   - normalize selected written forms for speech;
   - allow a human to review and edit the result.
2. **Text → long-form audio**
   - split reviewed text into complete sentence units;
   - group sentences into bounded-duration chunks;
   - condition SoulX-Podcast on reference audio and its transcript;
   - generate sentence audio;
   - assemble all generated segments into a final WAV file;
   - retain a segment manifest and allow sentence-level correction/regeneration.

The current workflow deliberately keeps a human review step between extraction and speech generation. The PDF-cleaning code is conservative because preserving the academic content is more important than aggressively rewriting it.

---

## 2. Current capabilities

### PDF/document processing

The repository provides two extraction paths.

**GROBID path (preferred for structured academic papers)**

The application can send a PDF to a running GROBID `processFulltextDocument` endpoint. GROBID returns TEI XML. The code parses that XML into a document representation containing the title, abstract, section headings, and paragraphs. Before rendering speech text, the cleaner can remove figures, tables, notes, bibliographic/footnote references, and residual inline citations.

**Local PyMuPDF fallback**

For environments where a GROBID service is not available, `PyMuPDF` is used to extract text blocks directly from born-digital PDFs. The fallback applies conservative heuristics to:

- remove repeated short page-edge blocks that are likely running headers/footers;
- remove bare page-number blocks;
- stop reading when a standalone `References`, `Bibliography`, `Works Cited`, or `Literature Cited` heading is encountered;
- remove likely short numeric footnote blocks when note removal is enabled;
- remove selected numeric and author-year inline citation patterns;
- repair common line-break hyphenation;
- normalize whitespace.

The local fallback is intended for selectable-text PDFs. OCR for scanned/image-only PDFs is **not implemented**.

### Speech-oriented normalization

The current normalizer is intentionally conservative. It performs transformations implemented in `soulxpodcast/document/speech_normalizer.py`, including:

- `Fig.` → `Figure`;
- `Figs.` → `Figures`;
- `Eq.` → `Equation`;
- `Eqs.` → `Equations`;
- `et al.` → `and colleagues`;
- whitespace normalization.

It does **not** attempt broad semantic rewriting of equations, mathematical notation, scientific notation, or arbitrary symbols.

### Long-form speech generation

The speech side is based on SoulX-Podcast. The application supports:

- reference-audio conditioning / zero-shot voice cloning;
- reference transcript input;
- long-form sentence splitting;
- duration estimation and chunk assignment;
- one- or two-speaker scripts using `S1` and `S2`;
- optional dialect prompt fields already supported by the underlying inference pipeline;
- chunked generation;
- assembly into a final audio file;
- configurable inter-segment gap;
- editable sentence/chunk table;
- sentence-level correction and selective regeneration;
- a review timeline;
- downloadable final WAV audio;
- downloadable segment manifest JSON.

Generated assembled audio is written at **24,000 Hz** by the current GUI/inference code.

---

## 3. System architecture

```text
Academic PDF
    |
    v
+-------------------------+
| Document ingestion      |
|                         |
| Auto / GROBID / PyMuPDF |
+-------------------------+
    |
    v
+-------------------------+
| Academic cleaning       |
| - references            |
| - footnotes/notes       |
| - citations             |
| - headers/footers       |
| - page numbers          |
+-------------------------+
    |
    v
+-------------------------+
| Speech normalization    |
+-------------------------+
    |
    v
Editable speech-ready text
    |
    | human review
    v
+-------------------------+
| Sentence segmentation   |
| and chunk assignment    |
+-------------------------+
    |
    v
+-------------------------+
| SoulX-Podcast inference |
| + reference voice       |
+-------------------------+
    |
    v
Sentence audio segments
    |
    v
+-------------------------+
| Audio assembly          |
| + JSON manifest         |
+-------------------------+
    |
    v
Final 24 kHz WAV
```

### Important design principle

The PDF stage and speech stage are separate. A PDF is first converted into **reviewable text**. The text is not sent blindly into the TTS model. This makes it possible to catch extraction errors, accidental citation retention, missing content, or undesirable pronunciation text before expensive speech generation.

---

## 4. Repository structure

The focused application uses the following main files and packages:

```text
cwcge-pdf2audio/
├── LICENSE
├── requirements.txt
├── readme.md
├── unified_webui.py
├── document_webui.py
├── colab_webui.py
├── cli/
│   ├── paper_to_text.py
│   └── tts.py
├── soulxpodcast/
│   ├── config.py
│   ├── document/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── grobid_client.py
│   │   ├── pdf_fallback.py
│   │   ├── pipeline.py
│   │   ├── speech_normalizer.py
│   │   └── tei_parser.py
│   ├── engine/
│   │   └── llm_engine.py
│   ├── models/
│   │   ├── soulxpodcast.py
│   │   └── modules/
│   │       ├── flow.py
│   │       ├── sampler.py
│   │       ├── hifigan.py
│   │       ├── flow_components/
│   │       └── hifigan_components/
│   └── utils/
│       ├── audio.py
│       ├── commons.py
│       ├── dataloader.py
│       ├── infer_utils.py
│       ├── parser.py
│       ├── segmented_workflow.py
│       └── text.py
└── tests/
    ├── test_document_ingestion.py
    └── test_segmented_workflow.py
```

### Entry points

`unified_webui.py` is the main user-facing entry point. It combines the document and speech interfaces in one Gradio application with two tabs.

`document_webui.py` is a lightweight development/testing interface for PDF → text only. It does not initialize the SoulX model and is therefore useful when working specifically on document extraction.

`colab_webui.py` contains the segmented long-form speech GUI used by the unified application. Despite the historical filename, the code is a normal Gradio application and is not architecturally dependent on Google Colab.

`cli/paper_to_text.py` provides a command-line GROBID-based PDF → text path.

`cli/tts.py` provides the lower-level command-line text → speech path.

---

## 5. Requirements

The checked-in `requirements.txt` currently declares:

```text
librosa
numpy
scipy
s3tokenizer
diffusers
torch==2.7.1
torchaudio==2.7.1
triton>=3.0.0
transformers==4.57.1
accelerate==1.10.1
onnxruntime
onnxruntime-gpu
einops
gradio
pymupdf
```

A CUDA-capable GPU is strongly recommended for SoulX speech generation. PDF → text testing with PyMuPDF does not require the speech model or a GPU.

The application expects the SoulX model directory to contain these files:

```text
soulxpodcast_config.json
flow.pt
hift.pt
```

By default the GUI searches for:

```text
pretrained_models/SoulX-Podcast-1.7B/
```

It can also receive an explicit model directory through `--model_path`.

---

## 6. Installation on a normal Linux machine

Python 3.11 is the safest target for the original SoulX dependency stack.

```bash
git clone https://github.com/JerryLee36/cwcge-pdf2audio.git
cd cwcge-pdf2audio

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the Hugging Face command-line tooling if `hf` is not already available:

```bash
pip install -U huggingface_hub
```

Download the model expected by the GUI:

```bash
hf download Soul-AILab/SoulX-Podcast-1.7B \
  --local-dir pretrained_models/SoulX-Podcast-1.7B
```

Then start the main interface:

```bash
python unified_webui.py \
  --model_path pretrained_models/SoulX-Podcast-1.7B
```

The default port is `7860` and the server binds to `0.0.0.0`.

If the model is already stored in the default `pretrained_models/SoulX-Podcast-1.7B` location, `--model_path` may be omitted.

---

## 7. Testing on Google Colab

Google Colab is supported as a **testing environment**, not as a requirement of the application architecture.

For the development branch used while the PDF pipeline is being integrated:

```python
!git clone -b feature/pdf-speech-ingestion https://github.com/JerryLee36/cwcge-pdf2audio.git
%cd cwcge-pdf2audio
!pip install -r requirements.txt
!pip install -U huggingface_hub
```

Download the SoulX model:

```python
!hf download Soul-AILab/SoulX-Podcast-1.7B \
    --local-dir pretrained_models/SoulX-Podcast-1.7B
```

Launch the unified GUI with a public Gradio link:

```python
!python unified_webui.py \
    --model_path pretrained_models/SoulX-Podcast-1.7B \
    --share
```

If you only want to test PDF extraction and do **not** want to download or initialize the TTS model:

```python
!python document_webui.py --share
```

This lightweight mode is particularly useful while improving citation, footnote, and reference cleaning.

---

## 8. Using the unified Web GUI

The unified interface contains two tabs.

### Tab 1 — PDF → Text

1. Upload an academic PDF.
2. Choose an extraction backend:
   - **Auto** — try GROBID first and fall back to PyMuPDF when GROBID is unavailable;
   - **Local (PyMuPDF)** — always use the local extractor;
   - **GROBID** — require the configured GROBID service.
3. Configure the available cleaning options:
   - keep/remove title;
   - keep/remove abstract;
   - keep/remove section headings;
   - remove/keep inline citations.
4. Click **Extract and clean text**.
5. Read the status message to see which backend was actually used.
6. Review the extracted text in the editable textbox.
7. Correct any extraction or cleaning mistakes manually.
8. Download the generated `.speech.txt` file if desired.

The document GUI stores generated text under:

```text
outputs/document_text/
```

with a timestamped `.speech.txt` filename.

### Tab 2 — Text → Audio

The current unified interface intentionally requires the reviewed text to be pasted into the **Initial Script** field. This preserves a human verification step between document extraction and voice generation.

For a single-reader lecture note, plain text is treated as speaker `S1`. The segmented workflow also understands explicit speaker tags such as:

```text
[S1]This paragraph is spoken by speaker one.
[S2]This paragraph is spoken by speaker two.
```

The current GUI supports `S1` and `S2`.

For each speaker used in the script, provide:

- prompt/reference audio;
- the exact prompt/reference transcript;
- an optional dialect prompt if that feature is required.

Then:

1. set the desired maximum chunk duration;
2. click **Split into sentence chunks**;
3. inspect the editable sentence/chunk table;
4. edit sentence text if necessary;
5. click **Apply sentence list edits** after table edits;
6. set the seed and inter-segment gap as desired;
7. click **Generate all chunks**;
8. listen to the assembled audio;
9. inspect the review timeline;
10. download the final audio and JSON manifest.

If one sentence is incorrect, select its sentence ID in the correction section, edit the text/speaker, and use **Regenerate selected sentence**. The application preserves already generated segments where possible and reassembles the final output.

---

## 9. PDF extraction backends in detail

### 9.1 GROBID

The GROBID client is implemented in `soulxpodcast/document/grobid_client.py` using Python's standard `urllib` modules. It posts the PDF as multipart form data to:

```text
<grobid-base-url>/api/processFulltextDocument
```

The default base URL is:

```text
http://localhost:8070
```

GROBID is **not** installed or started by this repository. If GROBID mode is selected, a compatible GROBID server must already be reachable at the configured URL.

The returned TEI XML is processed by:

```text
GrobidClient
    -> clean_tei_for_speech()
    -> parse_tei()
    -> _render_document()
    -> normalize_for_speech()
```

The structured path is preferable when accurate academic document structure is important.

### 9.2 PyMuPDF fallback

The local fallback is implemented in `soulxpodcast/document/pdf_fallback.py` and imports `fitz` from PyMuPDF.

It reads page text using:

```python
page.get_text("blocks")
```

and then processes block text using deterministic rules.

The fallback does not attempt OCR. A scanned PDF with no embedded/selectable text can therefore return no readable text, in which case the GUI reports that OCR is not implemented.

### 9.3 Citation cleaning

The cleaner contains conservative regular expressions for common patterns such as:

```text
[1]
[1, 3, 7]
[4-8]
(Smith, 2021)
(Smith et al., 2021; Jones, 2022)
```

Narrative year-only parentheses are also removed, so a form such as:

```text
Smith et al. (2021) demonstrated ...
```

can become speech text without the parenthetical year. The subsequent speech normalizer converts `et al.` to `and colleagues`.

Citation removal is heuristic. Users should review the extracted text before generating speech.

---

## 10. Long-form segmentation

Long lecture notes cannot be treated as one unbounded model input. The long-form workflow in `soulxpodcast/utils/segmented_workflow.py` therefore represents the script as sentence units.

Each sentence unit contains information including:

```text
sentence_id
speaker
text
chunk_id
estimate_seconds
```

Sentence boundaries include English and Chinese terminal punctuation. The splitter contains handling for common abbreviations and cases such as decimal points so that every period is not automatically treated as a sentence boundary.

Sentences are assigned to chunks according to estimated duration and the configured maximum chunk duration. A single sentence that itself exceeds the chunk limit must be shortened before generation.

This structure is also what makes sentence-level regeneration possible: generated audio is associated with stable sentence IDs rather than being treated as one opaque waveform.

---

## 11. Voice cloning and inference

The main inference preparation functions are in:

```text
soulxpodcast/utils/infer_utils.py
```

`initiate_model()` loads the SoulX configuration/model and creates the inference handler. `process_single_input()` prepares the target text and reference conditioning information, including prompt audio, prompt transcript, speaker information, and optional dialect prompt data.

The GUI then calls the SoulX model's long-form forward path and receives generated waveforms. Generated sentence waveforms are converted to NumPy arrays, associated with their sentence IDs, and assembled into the final output.

### Reference voice requirements

For every speaker used by the script, the GUI validates that both of these are present:

```text
reference/prompt audio
reference/prompt transcript
```

For the normal CW Chu College lecture-note use case, use a single `S1` speaker and provide an authorized reference recording of the intended lecturer/reader together with the transcript of that recording.

---

## 12. Output files

### Document stage

PDF extraction creates timestamped speech text under:

```text
outputs/document_text/
```

### Speech stage

The segmented GUI creates a run-specific output directory and writes:

```text
final_audio.wav
segments_manifest.json
```

The manifest records segment-level information used by the review workflow, including sentence/chunk identity and timing information after assembly.

Generated files under `outputs/` are runtime artifacts and should generally not be committed to source control.

---

## 13. Command-line PDF → text

The repository retains a small command-line document extractor:

```bash
python cli/paper_to_text.py lecture.pdf
```

By default it writes:

```text
lecture.speech.txt
```

Available options implemented by the CLI include:

```text
--output / -o
--grobid-url
--no-title
--no-abstract
--no-headings
--keep-citations
```

Example:

```bash
python cli/paper_to_text.py lecture.pdf \
  --output lecture.cleaned.txt \
  --grobid-url http://localhost:8070 \
  --no-abstract
```

Important: this CLI currently calls the GROBID pipeline directly. The **Auto → PyMuPDF fallback behavior is implemented in the document Web GUI**, not in `cli/paper_to_text.py`.

---

## 14. Command-line text → speech

`cli/tts.py` remains available for lower-level monologue inference. It accepts target text, reference audio, reference transcript, model path, output path, engine selection, and seed settings.

The target `--text` argument may be literal text or a path to a UTF-8 text file. The CLI constructs an `S1` input, calls the model's long-form forward path, concatenates returned waveforms, and writes a 24 kHz WAV file.

Consult the command's built-in help for the exact options in the checked-out version:

```bash
python cli/tts.py --help
```

---

## 15. Tests

Two focused test modules are retained:

```text
tests/test_document_ingestion.py
tests/test_segmented_workflow.py
```

The document tests cover TEI parsing/cleaning behavior. The segmented-workflow tests cover the long-form sentence/chunk and assembly utilities.

To run them, install `pytest` if it is not already present:

```bash
pip install pytest
pytest -q
```

The test suite does not replace manual testing with representative lecture-note PDFs. Academic PDFs vary substantially in layout and citation/footnote conventions.

---

## 16. Known limitations

The current code has the following important limitations:

- **No OCR.** Scanned/image-only PDFs are not supported by the local extractor.
- **PyMuPDF cleaning is heuristic.** It does not provide the same semantic structure as a scholarly parser.
- **GROBID is an external service.** This repository does not install or launch GROBID.
- **Citation removal is pattern based.** Unusual citation formats may remain, and unusual parenthetical academic text may require manual review.
- **Footnote detection in local mode is heuristic.** Complex page layouts can confuse block-level rules.
- **Mathematics is not comprehensively verbalized.** Equations and scientific notation are deliberately not aggressively rewritten.
- **The unified GUI currently uses a manual review/copy step between PDF text and the Initial Script field.** This is intentional in the current implementation.
- **The unified GUI initializes the speech model at application startup.** Use `document_webui.py` when testing PDF extraction without model initialization.
- **Current GUI speaker validation supports S1/S2.** The normal lecture-note use case is expected to use S1 only.
- **Long individual sentences can exceed the configured chunk limit.** Such sentences must be edited/shortened.
- **Voice quality depends on the underlying SoulX model and the quality/correctness of the reference audio and transcript.**

---

## 17. Development priorities

The current codebase is intentionally focused on the working PDF-to-reviewed-text and reviewed-text-to-audio pipeline. Logical future improvements include:

1. improve academic footnote detection for the local PDF path;
2. improve reference/citation removal against a representative CW Chu College document set;
3. add OCR only if scanned lecture notes become an actual requirement;
4. define and test explicit verbalization rules for mathematics and scientific notation;
5. optionally connect reviewed PDF text directly to the TTS script field while preserving an explicit human approval step;
6. add end-to-end regression fixtures using non-sensitive representative lecture-note documents.

These are future directions, not current features.

---

## 18. Responsible voice use

This project includes zero-shot voice cloning. Use reference voices only when their use is authorized for the intended educational purpose.

Do not use the software for unauthorized impersonation, fraud, deceptive audio, or other unlawful activity. For the intended CW Chu College workflow, the reference recording should be provided or approved for the production of the relevant educational audio.

---

## 19. Upstream technology and acknowledgement

The speech synthesis core in this repository is derived from **SoulX-Podcast**, whose model architecture and inference implementation provide the long-form speech generation and voice-conditioning capabilities used by this application.

The original SoulX-Podcast project describes support for long-form speech, zero-shot voice cloning, multilingual/dialect speech, and paralinguistic control. This application narrows that technology to the lecture-note PDF-to-audio workflow and adds the academic-document ingestion/cleaning layer and segmented review workflow used here.

The original repository also acknowledges FlashCosyVoice. The existing model implementation and its dependencies have been retained where they are required by SoulX inference.

When publishing research that relies on the underlying SoulX-Podcast model, consult and cite the original SoulX-Podcast work as appropriate.

---

## 20. License

The repository retains the existing **Apache License 2.0** in `LICENSE`.

The presence of application-specific modifications does not remove upstream license obligations. Review `LICENSE` before redistribution or deployment, and preserve required notices/attribution for upstream code and model components.

Model weights are downloaded separately and may have their own distribution/use terms. Users are responsible for checking the terms associated with the model weights they download.

---

## 21. Project context and contact

**Project:** cwcge-pdf2audio  
**Use:** audio production for lecture notes and academic materials associated with CW Chu College General Education units  
**Requested by:** Professor Kenneth Young  
**Guidance:** Professor Tan Lee and Paco Fung  
**Maintainer:** Yuzhe Li  
**Email:** yuzheli52@gmail.com

For implementation questions, PDF-cleaning failures, or reproducible bugs, include the backend used (`GROBID` or `PyMuPDF`), the relevant traceback/status message, and a minimal non-sensitive example when possible.
