"""Translation helpers (EN -> DE)."""

from __future__ import annotations

import re
from typing import Iterable, List

from .config import LFM_BATCH
from .models import build_chat_prompt, lfm_generate

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_EN_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_BAD_MARKERS = re.compile(r"(?i)\b(translate|translation|german|deutsch|english|labels?|explanations?|note|hinweis|anmerkung)\b")


def _split_sentences(text: str, max_chars: int = 260) -> List[str]:
    parts = [part.strip() for part in _SENT_SPLIT_RE.split(text.replace("\n", " ").strip()) if part.strip()]
    if not parts:
        return []
    buffer = ""
    output: List[str] = []
    for part in parts:
        if not buffer:
            buffer = part
            continue
        candidate = f"{buffer} {part}" if buffer else part
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            output.append(buffer)
            buffer = part
    if buffer:
        output.append(buffer)
    return output


def translate(text: str) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    prompts = [
        build_chat_prompt(
            [
                {
                    "role": "system",
                    "content": "You are a precise translation engine. Translate English into German without adding comments or extra explanations.",
                },
                {"role": "user", "content": sentence},
            ]
        )
        for sentence in sentences
    ]
    outputs = []
    for index in range(0, len(prompts), LFM_BATCH):
        batch = prompts[index:index + LFM_BATCH]
        outputs.extend(lfm_generate(batch))
    cleaned: List[str] = []
    for candidate in outputs:
        candidate = re.sub(r"(?i)^(assistant|system|user)\s*:\s*", "", candidate).strip()
        candidate = re.sub(r"(?im)^(german|deutsch|english|note|hinweis|anmerkung)\s*:\s*", "", candidate).strip()
        cleaned.append(candidate)
    final = " ".join(filter(None, cleaned)).strip()
    final = re.sub(r"\s+([,.;:!?])", r"\1", final)
    return final

