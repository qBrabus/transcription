"""Transcription (FR -> EN) using Canary."""

from __future__ import annotations

from typing import Iterable, List

from .models import load_canary


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
    model = load_canary()
    kwargs = dict(
        audio=list(paths),
        batch_size=batch_size,
        taskname="ast",
        source_lang="fr",
        target_lang="en",
        pnc="yes",
        num_workers=0,
        pretokenize=False,
        pin_memory=False,
    )
    try:
        outputs = model.transcribe(**kwargs)
    except TypeError:
        outputs = model.transcribe(audio=list(paths), batch_size=batch_size)
    if isinstance(outputs, list):
        return [_normalize_canary_output(item).strip() for item in outputs]
    if isinstance(outputs, str):
        return [outputs.strip()]
    return [str(outputs).strip()]

