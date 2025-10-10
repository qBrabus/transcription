"""Gradio interface for the transcription pipeline."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import gradio as gr

from .audio_utils import have_ffmpeg
from .pipeline import export_results, process


def _copy_to_temp(src: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="in_"))
    destination = temp_dir / src.name
    with src.open("rb") as src_file, destination.open("wb") as dst_file:
        shutil.copyfileobj(src_file, dst_file, length=1024 * 1024)
    return destination


def _extract_path(file_input) -> Path:
    if isinstance(file_input, str):
        return Path(file_input)
    if isinstance(file_input, dict):
        path = file_input.get("path") or file_input.get("name")
        if path:
            return Path(path)
    raise RuntimeError("Invalid file input.")


def process_file(file_input):
    try:
        if not have_ffmpeg():
            raise RuntimeError("ffmpeg is required on PATH to decode audio/video files.")
        source = _extract_path(file_input)
        if not source.exists():
            raise RuntimeError(f"File not found: {source}")
        safe_copy = _copy_to_temp(source)
        transcript_en, transcript_de, data = process(safe_copy)
        json_path, en_path, de_path = export_results(transcript_en, transcript_de, data)
        return transcript_en, transcript_de, str(json_path), str(en_path), str(de_path)
    except Exception as exc:
        return f"Error: {exc}", "", None, None, None


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="FR ➜ EN (diarization) ➜ DE") as demo:
        gr.Markdown(
            """
            ### French audio ➜ English transcription ➜ German translation
            * GPU-first pipeline with diarization, automatic aliasing and structured exports.
            * The system automatically detects names or roles directly from the transcript using an LLM.
            """
        )
        with gr.Row():
            file_input = gr.File(
                label="Upload French audio or video",
                file_count="single",
                file_types=["audio", "video"],
            )
        run_button = gr.Button("Process")
        with gr.Row():
            transcript_en = gr.Textbox(label="English transcript", lines=16)
            transcript_de = gr.Textbox(label="German translation", lines=16)
        with gr.Row():
            json_file = gr.File(label="JSON export")
            en_file = gr.File(label="EN transcript")
            de_file = gr.File(label="DE translation")
        run_button.click(
            fn=process_file,
            inputs=file_input,
            outputs=[transcript_en, transcript_de, json_file, en_file, de_file],
        )
    return demo

