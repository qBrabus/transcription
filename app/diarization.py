"""Diarization helpers built around pyannote pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from pyannote.audio import Pipeline

from .audio_utils import Segment, load_audio
from .config import (
    DEVICE,
    DIARIZATION_PRIMARY,
    DIARIZATION_SECONDARY,
    MERGE_GAP_INPUT,
    MIN_SILENCE,
    MIN_SPEECH,
    PYANNOTE_CACHE,
)


@dataclass
class DiarizationResult:
    segments: List[Segment]
    score: float


def _speaker_label(raw: str) -> str:
    label = str(raw)
    if not label.upper().startswith("SPEAKER_"):
        label = f"SPEAKER_{label}"
    return label


def _merge_segments(segments: List[Segment], gap: float = MERGE_GAP_INPUT) -> List[Segment]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda seg: (seg.start, seg.end))
    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.speaker == prev.speaker and (seg.start - prev.end) <= gap:
            merged[-1] = Segment(prev.speaker, prev.start, max(prev.end, seg.end))
        else:
            merged.append(seg)
    return merged


def _absorb_micro_segments(segments: List[Segment]) -> List[Segment]:
    if not segments:
        return []
    output: List[Segment] = []
    current = segments[0]
    for segment in segments[1:]:
        gap = segment.start - current.end
        if gap > 0 and gap < MIN_SILENCE and segment.speaker == current.speaker:
            current = Segment(current.speaker, current.start, segment.end)
            continue
        if current.duration < MIN_SPEECH and segment.speaker == current.speaker and gap <= MERGE_GAP_INPUT:
            current = Segment(current.speaker, current.start, max(current.end, segment.end))
            continue
        output.append(current)
        current = segment
    if current.duration < MIN_SPEECH and output and output[-1].speaker == current.speaker and (current.start - output[-1].end) <= MERGE_GAP_INPUT:
        output[-1] = Segment(output[-1].speaker, output[-1].start, max(output[-1].end, current.end))
    else:
        output.append(current)
    return output


def _median_filter_labels(segments: List[Segment], max_island: float = 0.8) -> List[Segment]:
    if len(segments) < 3:
        return segments
    segments = segments[:]
    for idx in range(1, len(segments) - 1):
        prev, current, nxt = segments[idx - 1], segments[idx], segments[idx + 1]
        if prev.speaker == nxt.speaker and current.speaker != prev.speaker and (current.end - current.start) <= max_island:
            segments[idx] = Segment(prev.speaker, current.start, current.end)
    return _merge_segments(segments)


def _jitter_score(segments: List[Segment]) -> float:
    if not segments:
        return 1e9
    total_duration = segments[-1].end - segments[0].start
    changes = sum(1 for index in range(1, len(segments)) if segments[index].speaker != segments[index - 1].speaker)
    average_duration = sum(seg.duration for seg in segments) / max(1, len(segments))
    changes_per_minute = changes / max(1e-6, total_duration / 60.0)
    return changes_per_minute - 0.5 * average_duration


def _run_pipeline(pipeline: Pipeline, audio_path) -> List[Segment]:  # type: ignore[override]
    waveform = load_audio(audio_path)
    diarization = pipeline(waveform)
    segments: List[Segment] = []
    if hasattr(diarization, "speaker_diarization"):
        for turn, speaker in diarization.speaker_diarization:
            segments.append(Segment(_speaker_label(speaker), float(turn.start), float(turn.end)))
    elif hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(Segment(_speaker_label(speaker), float(turn.start), float(turn.end)))
    elif hasattr(diarization, "annotation") and hasattr(diarization.annotation, "itertracks"):
        for turn, _, speaker in diarization.annotation.itertracks(yield_label=True):
            segments.append(Segment(_speaker_label(speaker), float(turn.start), float(turn.end)))
    else:
        raise RuntimeError("Unsupported diarization output structure.")
    return sorted(segments, key=lambda seg: (seg.start, seg.end))


def _refine(segments: List[Segment]) -> List[Segment]:
    refined = _absorb_micro_segments(segments)
    refined = _median_filter_labels(refined)
    refined = _merge_segments(refined)
    return refined


def diarize(pipeline: Pipeline, audio_path) -> DiarizationResult:
    segments = _run_pipeline(pipeline, audio_path)
    segments = _refine(segments)
    score = _jitter_score(segments)
    return DiarizationResult(segments, score)


def make_pipeline(repo_id: str) -> Pipeline:
    pipeline = Pipeline.from_pretrained(repo_id, cache_dir=str(PYANNOTE_CACHE))
    pipeline.to(DEVICE)
    return pipeline


def choose_best_diarization(audio_path) -> List[Segment]:  # type: ignore[override]
    primary = diarize(make_pipeline(DIARIZATION_PRIMARY), audio_path)
    try:
        secondary_pipeline = make_pipeline(DIARIZATION_SECONDARY)
        secondary = diarize(secondary_pipeline, audio_path)
    except Exception:
        secondary = DiarizationResult([], 1e9)
    return primary.segments if primary.score <= secondary.score else secondary.segments

