import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from soulxpodcast.utils.infer_utils import initiate_model, process_single_input
from soulxpodcast.utils.segmented_workflow import (
    assemble_audio_and_manifest,
    normalize_sentence_table_rows,
    replace_segment_audio,
    split_script_into_sentence_units,
    table_to_units,
    unit_to_dict,
)


def _to_rows(table_value: Any) -> List[Dict[str, Any]]:
    return normalize_sentence_table_rows(table_value)


def _segments_from_units(units: Sequence[Dict[str, Any]], previous_segments: Sequence[Dict[str, Any]] | None = None):
    old_by_id = {int(item["sentence_id"]): item for item in (previous_segments or [])}
    segments = []
    for unit in units:
        sid = int(unit["sentence_id"])
        current = {
            "sentence_id": sid,
            "chunk_id": int(unit["chunk_id"]),
            "speaker": unit["speaker"],
            "text": unit["text"],
            "audio": None,
        }
        previous = old_by_id.get(sid)
        if previous and previous.get("text") == unit["text"] and previous.get("speaker") == unit["speaker"]:
            current["audio"] = previous.get("audio")
        segments.append(current)
    return segments


def _validate_prompt_inputs(units: Sequence[Dict[str, Any]], spk1_audio, spk1_text, spk2_audio, spk2_text):
    speakers = {item["speaker"] for item in units}
    unsupported = {speaker for speaker in speakers if speaker not in {"S1", "S2"}}
    if unsupported:
        raise ValueError(f"This GUI currently supports S1/S2 only. Found unsupported speakers: {sorted(unsupported)}")

    if "S1" in speakers and (not spk1_audio or not str(spk1_text).strip()):
        raise ValueError("S1 is used in script. Please provide both S1 prompt audio and prompt text.")

    if "S2" in speakers and (not spk2_audio or not str(spk2_text).strip()):
        raise ValueError("S2 is used in script. Please provide both S2 prompt audio and prompt text.")


def _prompt_lists(speakers: Sequence[str], spk1_audio, spk1_text, spk1_dialect, spk2_audio, spk2_text, spk2_dialect):
    if speakers == ["S1"]:
        return [spk1_audio], [spk1_text], [spk1_dialect], bool(str(spk1_dialect).strip())

    # S1/S2 mode keeps original speaker indexing for inference
    return (
        [spk1_audio, spk2_audio],
        [spk1_text, spk2_text],
        [spk1_dialect, spk2_dialect],
        bool(str(spk1_dialect).strip() or str(spk2_dialect).strip()),
    )


def _chunk_map_from_units(units: Sequence[Dict[str, Any]]) -> List[List[int]]:
    grouped: Dict[int, List[int]] = {}
    for item in units:
        grouped.setdefault(int(item["chunk_id"]), []).append(int(item["sentence_id"]))
    return [grouped[key] for key in sorted(grouped.keys())]


def _units_dict_to_table(units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "sentence_id": int(item["sentence_id"]),
            "chunk_id": int(item["chunk_id"]),
            "speaker": str(item["speaker"]),
            "text": str(item["text"]),
            "estimate_seconds": round(float(item.get("estimate_seconds", 0.0)), 2),
        }
        for item in units
    ]


def _sentence_dicts_to_rows(units: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    return [
        [
            int(item["sentence_id"]),
            int(item["chunk_id"]),
            str(item["speaker"]),
            str(item["text"]),
            round(float(item.get("estimate_seconds", 0.0)), 2),
        ]
        for item in units
    ]


def _review_dicts_to_rows(segments: Sequence[Dict[str, Any]]) -> List[List[Any]]:
    return [
        [
            int(item["sentence_id"]),
            int(item.get("chunk_id", -1)),
            str(item["speaker"]),
            str(item["text"]),
            int(item.get("start_sample", 0)),
            int(item.get("end_sample", 0)),
            round(float(item.get("start_seconds", 0.0)), 6),
            round(float(item.get("end_seconds", 0.0)), 6),
            round(float(item.get("duration_seconds", 0.0)), 6),
        ]
        for item in segments
    ]


def _save_outputs(work_dir: Path, audio: np.ndarray, manifest: Dict[str, Any]) -> Tuple[str, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "final_audio.wav"
    manifest_path = work_dir / "segments_manifest.json"
    sf.write(str(audio_path), audio, 24000)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return str(audio_path), str(manifest_path)


def create_app(model_path: str, llm_engine: str, fp16_flow: bool, seed: int):
    model, dataset = initiate_model(seed, model_path, llm_engine, fp16_flow)
    output_dir = Path("outputs") / "colab_gui" / datetime.now().strftime("%Y%m%d-%H%M%S")

    def split_script(script_text: str, max_chunk_seconds: float, state: Dict[str, Any]):
        state = state or {}
        try:
            units = split_script_into_sentence_units(script_text, max_chunk_seconds=max_chunk_seconds, default_speaker="S1")
            units_dict = [unit_to_dict(unit) for unit in units]
            previous_segments = state.get("segments", [])
            segments = _segments_from_units(units_dict, previous_segments=previous_segments)

            new_state = {
                "units": units_dict,
                "segments": segments,
                "max_chunk_seconds": max_chunk_seconds,
            }
            sentence_choices = [str(item["sentence_id"]) for item in units_dict]
            return (
                _sentence_dicts_to_rows(units_dict),
                gr.update(choices=sentence_choices, value=sentence_choices[0] if sentence_choices else None),
                f"Prepared {len(units_dict)} sentence(s) in {len(_chunk_map_from_units(units_dict))} chunk(s).",
                new_state,
            )
        except Exception as exc:
            return [], gr.update(choices=[], value=None), f"Split failed: {exc}", state

    def apply_table_edits(table_value: Any, state: Dict[str, Any]):
        state = state or {}
        try:
            rows = _to_rows(table_value)
            if not rows:
                raise ValueError("No sentence rows to apply")
            units = table_to_units(rows)

            max_chunk_seconds = float(state.get("max_chunk_seconds", 110.0))
            current_chunk = 0
            chunk_seconds = 0.0
            for unit in units:
                if unit.estimate_seconds > max_chunk_seconds:
                    raise ValueError(
                        f"Sentence {unit.sentence_id} exceeds chunk limit. Please shorten it."
                    )
                if chunk_seconds and chunk_seconds + unit.estimate_seconds > max_chunk_seconds:
                    current_chunk += 1
                    chunk_seconds = 0.0
                unit.chunk_id = current_chunk
                chunk_seconds += unit.estimate_seconds

            units_dict = [unit_to_dict(unit) for unit in units]
            segments = _segments_from_units(units_dict, previous_segments=state.get("segments", []))

            new_state = {
                "units": units_dict,
                "segments": segments,
                "max_chunk_seconds": max_chunk_seconds,
            }
            sentence_choices = [str(item["sentence_id"]) for item in units_dict]
            return (
                _sentence_dicts_to_rows(units_dict),
                gr.update(choices=sentence_choices, value=sentence_choices[0] if sentence_choices else None),
                "Edits applied.",
                new_state,
            )
        except Exception as exc:
            return table_value, gr.update(), f"Apply edits failed: {exc}", state

    def _generate_sentence_batch(
        batch_units: Sequence[Dict[str, Any]],
        spk1_prompt_audio,
        spk1_prompt_text,
        spk1_dialect_prompt,
        spk2_prompt_audio,
        spk2_prompt_text,
        spk2_dialect_prompt,
        seed_value: int,
    ) -> List[Tuple[int, np.ndarray]]:
        if not batch_units:
            return []

        torch.manual_seed(int(seed_value))
        np.random.seed(int(seed_value))

        speakers_in_batch = sorted({item["speaker"] for item in batch_units})
        prompt_wavs, prompt_texts, dialect_prompts, use_dialect = _prompt_lists(
            speakers_in_batch,
            spk1_prompt_audio,
            spk1_prompt_text,
            spk1_dialect_prompt,
            spk2_prompt_audio,
            spk2_prompt_text,
            spk2_dialect_prompt,
        )

        text_list = [f"[{item['speaker']}]{item['text']}" for item in batch_units]
        data = process_single_input(
            dataset,
            text_list,
            prompt_wavs,
            prompt_texts,
            use_dialect,
            dialect_prompts,
        )
        results_dict = model.forward_longform(**data)
        wavs = [wav.cpu().squeeze(0).numpy().astype(np.float32) for wav in results_dict["generated_wavs"]]
        if len(wavs) != len(batch_units):
            raise RuntimeError("Model output size mismatch with sentence batch")
        return [(int(batch_units[i]["sentence_id"]), wavs[i]) for i in range(len(batch_units))]

    def _reassemble_from_state(current_state: Dict[str, Any]):
        segments = current_state.get("segments", [])
        if not segments:
            raise ValueError("No segments in state")
        final_audio, manifest = assemble_audio_and_manifest(segments)
        audio_path, manifest_path = _save_outputs(output_dir, final_audio, manifest)
        review_rows = manifest["segments"]
        return (
            (24000, final_audio),
            review_rows,
            audio_path,
            manifest_path,
            manifest,
        )

    def generate_all(
        table_value,
        spk1_prompt_audio,
        spk1_prompt_text,
        spk1_dialect_prompt,
        spk2_prompt_audio,
        spk2_prompt_text,
        spk2_dialect_prompt,
        seed_value,
        state,
    ):
        state = state or {}
        try:
            rows = _to_rows(table_value)
            if rows:
                units = [unit_to_dict(item) for item in table_to_units(rows)]
            else:
                units = state.get("units", [])
            if not units:
                raise ValueError("Please split script first.")

            _validate_prompt_inputs(units, spk1_prompt_audio, spk1_prompt_text, spk2_prompt_audio, spk2_prompt_text)
            segments = _segments_from_units(units, previous_segments=state.get("segments", []))
            chunk_map = _chunk_map_from_units(units)
            by_sentence_id = {int(item["sentence_id"]): item for item in units}

            for sentence_ids in chunk_map:
                batch_units = [by_sentence_id[sid] for sid in sentence_ids]
                generated = _generate_sentence_batch(
                    batch_units,
                    spk1_prompt_audio,
                    spk1_prompt_text,
                    spk1_dialect_prompt,
                    spk2_prompt_audio,
                    spk2_prompt_text,
                    spk2_dialect_prompt,
                    int(seed_value),
                )
                for sentence_id, wav in generated:
                    for idx, segment in enumerate(segments):
                        if int(segment["sentence_id"]) == int(sentence_id):
                            segments[idx] = dict(segment)
                            segments[idx]["audio"] = wav
                            break

            new_state = {
                "units": units,
                "segments": segments,
                "max_chunk_seconds": float(state.get("max_chunk_seconds", 110.0)),
            }
            full_audio, review_rows, audio_path, manifest_path, _ = _reassemble_from_state(new_state)
            return full_audio, _review_dicts_to_rows(review_rows), audio_path, manifest_path, "Generation complete.", new_state
        except Exception as exc:
            return None, [], None, None, f"Generation failed: {exc}", state

    def fill_selected_sentence(sentence_id: str, state: Dict[str, Any]):
        state = state or {}
        units = state.get("units", [])
        for item in units:
            if str(item["sentence_id"]) == str(sentence_id):
                return item["text"], item["speaker"]
        return "", "S1"

    def regenerate_selected(
        table_value,
        sentence_id,
        replacement_text,
        replacement_speaker,
        spk1_prompt_audio,
        spk1_prompt_text,
        spk1_dialect_prompt,
        spk2_prompt_audio,
        spk2_prompt_text,
        spk2_dialect_prompt,
        seed_value,
        state,
    ):
        state = state or {}
        try:
            rows = _to_rows(table_value)
            units = [unit_to_dict(item) for item in table_to_units(rows)] if rows else state.get("units", [])
            if not units:
                raise ValueError("No segments available")

            sid = int(sentence_id)
            updated_units = []
            found = False
            for item in units:
                cloned = dict(item)
                if int(cloned["sentence_id"]) == sid:
                    found = True
                    if str(replacement_text).strip():
                        cloned["text"] = str(replacement_text).strip()
                    if str(replacement_speaker).strip():
                        cloned["speaker"] = str(replacement_speaker).strip().upper()
                updated_units.append(cloned)
            if not found:
                raise ValueError(f"sentence_id={sid} not found")

            _validate_prompt_inputs(updated_units, spk1_prompt_audio, spk1_prompt_text, spk2_prompt_audio, spk2_prompt_text)
            segments = _segments_from_units(updated_units, previous_segments=state.get("segments", []))

            target_unit = [item for item in updated_units if int(item["sentence_id"]) == sid][0]
            generated = _generate_sentence_batch(
                [target_unit],
                spk1_prompt_audio,
                spk1_prompt_text,
                spk1_dialect_prompt,
                spk2_prompt_audio,
                spk2_prompt_text,
                spk2_dialect_prompt,
                int(seed_value),
            )
            _, new_audio = generated[0]
            segments = replace_segment_audio(segments, sid, new_audio)

            new_state = {
                "units": updated_units,
                "segments": segments,
                "max_chunk_seconds": float(state.get("max_chunk_seconds", 110.0)),
            }
            full_audio, review_rows, audio_path, manifest_path, _ = _reassemble_from_state(new_state)

            updated_table = _sentence_dicts_to_rows(updated_units)
            return updated_table, full_audio, _review_dicts_to_rows(review_rows), audio_path, manifest_path, "Selected sentence regenerated.", new_state
        except Exception as exc:
            return table_value, None, [], None, None, f"Selective regeneration failed: {exc}", state

    with gr.Blocks(title="SoulX-Podcast Colab GUI") as app:
        gr.Markdown("## SoulX-Podcast Colab GUI\nScript editing, chunked generation, review, and sentence-level correction.")

        state = gr.State({"units": [], "segments": [], "max_chunk_seconds": 110.0})

        with gr.Row():
            seed_input = gr.Number(label="Seed", value=seed, precision=0)
            max_chunk_seconds = gr.Number(label="Max chunk seconds", value=110.0, precision=1)

        with gr.Row():
            with gr.Column():
                spk1_prompt_audio = gr.Audio(label="S1 Prompt Audio", type="filepath")
                spk1_prompt_text = gr.Textbox(label="S1 Prompt Text", lines=2)
                spk1_dialect_prompt = gr.Textbox(label="S1 Dialect Prompt (optional)", lines=2)
            with gr.Column():
                spk2_prompt_audio = gr.Audio(label="S2 Prompt Audio (optional)", type="filepath")
                spk2_prompt_text = gr.Textbox(label="S2 Prompt Text (optional)", lines=2)
                spk2_dialect_prompt = gr.Textbox(label="S2 Dialect Prompt (optional)", lines=2)

        script_input = gr.Textbox(label="Initial Script", lines=12, placeholder="Use plain text or [S1]/[S2] tagged script.")

        with gr.Row():
            split_btn = gr.Button("Split into sentence chunks", variant="secondary")
            apply_edits_btn = gr.Button("Apply sentence list edits", variant="secondary")
            generate_all_btn = gr.Button("Generate all chunks", variant="primary")

        sentence_table = gr.Dataframe(
            headers=["sentence_id", "chunk_id", "speaker", "text", "estimate_seconds"],
            datatype=["number", "number", "str", "str", "number"],
            row_count=(0, "dynamic"),
            col_count=(5, "fixed"),
            interactive=True,
            label="Sentence/chunk list (editable)",
        )

        status_box = gr.Textbox(label="Status", interactive=False)
        final_audio = gr.Audio(label="Assembled Audio")
        review_table = gr.Dataframe(
            headers=[
                "sentence_id", "chunk_id", "speaker", "text", "start_sample", "end_sample",
                "start_seconds", "end_seconds", "duration_seconds",
            ],
            interactive=False,
            label="Review timeline",
        )

        with gr.Row():
            final_audio_file = gr.File(label="Download final audio")
            final_manifest_file = gr.File(label="Download manifest JSON")

        gr.Markdown("### Sentence-level correction")
        with gr.Row():
            sentence_selector = gr.Dropdown(label="Sentence ID", choices=[])
            correction_speaker = gr.Dropdown(label="Speaker", choices=["S1", "S2"], value="S1")
        correction_text = gr.Textbox(label="Corrected sentence text", lines=3)
        regenerate_selected_btn = gr.Button("Regenerate selected sentence", variant="primary")

        split_btn.click(
            fn=split_script,
            inputs=[script_input, max_chunk_seconds, state],
            outputs=[sentence_table, sentence_selector, status_box, state],
        )

        apply_edits_btn.click(
            fn=apply_table_edits,
            inputs=[sentence_table, state],
            outputs=[sentence_table, sentence_selector, status_box, state],
        )

        generate_all_btn.click(
            fn=generate_all,
            inputs=[
                sentence_table,
                spk1_prompt_audio,
                spk1_prompt_text,
                spk1_dialect_prompt,
                spk2_prompt_audio,
                spk2_prompt_text,
                spk2_dialect_prompt,
                seed_input,
                state,
            ],
            outputs=[final_audio, review_table, final_audio_file, final_manifest_file, status_box, state],
        )

        sentence_selector.change(
            fn=fill_selected_sentence,
            inputs=[sentence_selector, state],
            outputs=[correction_text, correction_speaker],
        )

        regenerate_selected_btn.click(
            fn=regenerate_selected,
            inputs=[
                sentence_table,
                sentence_selector,
                correction_text,
                correction_speaker,
                spk1_prompt_audio,
                spk1_prompt_text,
                spk1_dialect_prompt,
                spk2_prompt_audio,
                spk2_prompt_text,
                spk2_dialect_prompt,
                seed_input,
                state,
            ],
            outputs=[sentence_table, final_audio, review_table, final_audio_file, final_manifest_file, status_box, state],
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="SoulX-Podcast Colab-compatible web GUI")
    parser.add_argument("--model_path", required=True, type=str, help="Path to SoulX-Podcast model directory")
    parser.add_argument("--llm_engine", default="hf", choices=["hf", "vllm"], help="Inference engine")
    parser.add_argument("--fp16_flow", action="store_true", help="Enable fp16 flow")
    parser.add_argument("--seed", type=int, default=1988, help="Seed for generation")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", default=True, help="Enable Gradio share link (recommended for Colab)")
    parser.add_argument("--no-share", dest="share", action="store_false", help="Disable Gradio share link")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    app = create_app(args.model_path, args.llm_engine, args.fp16_flow, args.seed)
    app.queue().launch(share=args.share, server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()