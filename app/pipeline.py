"""End-to-end processing pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from . import config
from .audio_utils import AudioChunk, Segment, build_audio_chunks
from .diarization import choose_best_diarization
from .logging_utils import setup_logging
from .models import ensure_models
from .role_assignment import request_aliases
from .text_utils import format_transcript, merge_adjacent, stitch_overlapping
from .transcription import canary_transcribe
from .translation import translate

LOGGER = setup_logging()


def _aggregate_by_segment(chunks: List[AudioChunk], english_chunks: List[str], num_segments: int) -> List[str]:
    per_segment: Dict[int, List[str]] = {}
    for chunk, text in zip(chunks, english_chunks):
        if text:
            per_segment.setdefault(chunk.segment_index, []).append(text.strip())
    aggregated: List[str] = []
    for index in range(num_segments):
        parts = per_segment.get(index, [])
        aggregated.append(stitch_overlapping(parts))
    return aggregated


def _build_results(segments: List[Segment], english_texts: List[str], german_texts: List[str], aliases: Dict[str, str]) -> List[Dict]:
    results: List[Dict] = []
    for segment, english, german in zip(segments, english_texts, german_texts):
        label = aliases.get(segment.speaker, segment.speaker)
        results.append(
            {
                "speaker": segment.speaker,
                "label": label,
                "start": segment.start,
                "end": segment.end,
                "text_en": english,
                "text_de": german,
            }
        )
    return results


def process(audio_file: Path) -> Tuple[str, str, List[Dict]]:
    ensure_models()
    segments = choose_best_diarization(audio_file)
    if not segments:
        raise RuntimeError("No speech segments detected.")

    chunks, temp_dir = build_audio_chunks(audio_file, segments)
    try:
        english_chunks = canary_transcribe([str(chunk.path) for chunk in chunks], batch_size=config.CANARY_BATCH)
        english_segments = _aggregate_by_segment(chunks, english_chunks, len(segments))
        turns = [(segment.speaker, text) for segment, text in zip(segments, english_segments)]
        aliases = request_aliases(turns)
        german_segments: List[str] = []
        for index in tqdm(range(0, len(english_segments), config.TRANSLATION_BATCH), desc="EN->DE", unit="seg"):
            block = english_segments[index:index + config.TRANSLATION_BATCH]
            german_segments.extend([translate(text) if text else "" for text in block])
        results = _build_results(segments, english_segments, german_segments, aliases)
        fused_results = merge_adjacent(results, config.MERGE_GAP_OUTPUT)
        transcript_en = format_transcript(fused_results, "text_en")
        transcript_de = format_transcript(fused_results, "text_de")
        return transcript_en, transcript_de, fused_results
    finally:
        temp_dir.cleanup()


def export_results(transcript_en: str, transcript_de: str, data: List[Dict]) -> Tuple[Path, Path, Path]:
    output_dir = Path(tempfile.mkdtemp(prefix="out_"))
    json_path = output_dir / "transcripts.json"
    en_path = output_dir / "transcript_en.txt"
    de_path = output_dir / "transcript_de.txt"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=config.JSON_INDENT), encoding="utf-8")
    en_path.write_text(transcript_en, encoding="utf-8")
    de_path.write_text(transcript_de, encoding="utf-8")
    return json_path, en_path, de_path

