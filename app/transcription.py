"""Transcription (FR -> EN) using Canary."""

from __future__ import annotations

from typing import Iterable, List

from .models import canary_transcribe_api


def _normalize_canary_output(item) -> str:
    if isinstance(item, dict):
        for key in ("pred_text", "text", "translation_text", "answer"):
            value = item.get(key)
            if isinstance(value, str):
                return value
        return ""
    value = str(item)
    if "text=" in value:
        try:
            return value.split("text=", 1)[1].split("'", 1)[1].rsplit("'", 1)[0]
        except Exception:
            return value
    return value


def canary_transcribe(paths: Iterable[str], batch_size: int = 6) -> List[str]:
    del batch_size  # The API handles batching internally.
    outputs = canary_transcribe_api(list(paths), language="fr", translate=True)
    return [_normalize_canary_output(item).strip() for item in outputs]

