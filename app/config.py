"""Global configuration for the transcription pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("TRANSCRIPTION_CUDA_VISIBLE_DEVICES", "0"))
ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = Path(os.environ.get("TRANSCRIPTION_BASE_DIR", ROOT_DIR))
MODELS_DIR = Path(os.environ.get("TRANSCRIPTION_MODELS_DIR", BASE_DIR / "models"))
CACHE_DIR = Path(os.environ.get("TRANSCRIPTION_CACHE_DIR", MODELS_DIR / "cache"))
GRADIO_TMP = Path(os.environ.get("TRANSCRIPTION_GRADIO_TMP", BASE_DIR / "gradio_tmp"))
PYANNOTE_CACHE = Path(os.environ.get("TRANSCRIPTION_PYANNOTE_CACHE", MODELS_DIR / "pyannote"))

for directory in (MODELS_DIR, CACHE_DIR, GRADIO_TMP, PYANNOTE_CACHE):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR))
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TMP))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BF16_AVAILABLE = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

torch.set_grad_enabled(False)
torch.set_num_threads(1)
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

TARGET_SR = 16_000

DIARIZATION_PRIMARY = "pyannote/speaker-diarization-3.1"
DIARIZATION_SECONDARY = "pyannote/speaker-diarization-community-1"

CANARY_REPO = "nvidia/canary-1b-v2"
CANARY_FILENAME = "canary-1b-v2.nemo"

LFM_REPO = "LiquidAI/LFM2-2.6B"

TRANSLATION_BATCH = 16
LFM_BATCH = 8
CANARY_BATCH = 6

MAX_CANARY_WINDOW = 35.0
SUBWIN_OVERLAP = 1.0
PAD_BEFORE = 0.25
PAD_AFTER = 0.25

MERGE_GAP_INPUT = 0.9
MERGE_GAP_OUTPUT = 2.0
MIN_SPEECH = 0.7
MIN_SILENCE = 0.25

MAX_NEW_TOKENS = 128

JSON_INDENT = 2

