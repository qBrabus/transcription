"""Utility helpers for working with transcripts."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from .audio_utils import seconds_to_timestamp

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")


def word_tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")


def stitch_overlapping(parts: Sequence[str]) -> str:
    if not parts:
        return ""
    output_tokens = word_tokens(parts[0])
    for part in parts[1:]:
        tokens = word_tokens(part)
        if not tokens:
            continue
        max_overlap = min(8, len(output_tokens), len(tokens))
        overlap_size = 0
        for size in range(max_overlap, 1, -1):
            if output_tokens[-size:] == tokens[:size]:
                overlap_size = size
                break
        if overlap_size:
            output_tokens.extend(tokens[overlap_size:])
        else:
            if not output_tokens or tokens[0] != output_tokens[-1]:
                output_tokens.extend(tokens)
    text = " ".join(output_tokens)
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()


def merge_adjacent(entries: List[Dict], gap: float) -> List[Dict]:
    if not entries:
        return entries
    entries = sorted(entries, key=lambda item: (item["start"], item["end"]))
    merged: List[Dict] = [entries[0].copy()]
    for entry in entries[1:]:
        current = merged[-1]
        if (
            entry["label"] == current["label"]
            and (entry["start"] - current["end"]) <= gap
        ):
            gap_seconds = max(0.0, entry["start"] - current["end"])
            current["end"] = max(current["end"], entry["end"])
            for key in ("text_en", "text_de"):
                if key not in entry:
                    continue
                prev_text = current.get(key, "").rstrip()
                new_text = entry.get(key, "").lstrip()
                if not prev_text:
                    current[key] = new_text
                elif not new_text:
                    continue
                elif gap_seconds >= 1.0:
                    current[key] = f"{prev_text}\n\n{new_text}"
                else:
                    separator = " " if prev_text.endswith(tuple(".!?")) else ", "
                    current[key] = f"{prev_text}{separator}{new_text}"
        else:
            merged.append(entry.copy())
    return merged


def format_transcript(entries: Iterable[Dict], key: str) -> str:
    lines: List[str] = []
    for entry in entries:
        text = (entry.get(key) or "").strip()
        if not text:
            continue
        timestamp = f"[{seconds_to_timestamp(entry['start'])} - {seconds_to_timestamp(entry['end'])}] {entry['label']}:"
        lines.append(f"{timestamp}\n{text}\n")
    return "\n".join(lines).strip()

