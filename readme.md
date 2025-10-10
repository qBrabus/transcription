# Transcription Pipeline

This project provides a GPU-first Gradio application that performs:

1. **French audio diarization** using pyannote.
2. **Automatic speech translation** from French speech to English text with NVIDIA Canary.
3. **Role/name attribution** using the LFM model without relying on predefined role lists.
4. **High quality German translation** using the same LFM model.
5. **Structured exports** including JSON and plain text transcripts.

## Features

- Fully modular Python package under `app/` for easier maintenance.
- Automatic model caching inside the repository (configurable via environment variables).
- Speaker aliasing backed by LLM reasoning with evidence validation.
- Robust chunking and overlap handling for arbitrarily long audio files.
- GPU-friendly defaults (automatic bf16/float16 when available).
- Clean transcript formatting with timestamps.

## Usage

1. Export a valid Hugging Face token: `export HUGGINGFACE_TOKEN=...`.
2. Install dependencies listed in your environment (pyannote, nemo, transformers, gradio, etc.).
3. Launch the Gradio interface: `python main.py`.
4. Upload a French audio/video file and wait for diarization, transcription, aliasing and translation.

All intermediate artefacts are written to temporary directories that are automatically cleaned.

