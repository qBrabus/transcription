"""Audio utilities for preprocessing and chunking."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
import soxr
import torch

from .config import (
    MAX_CANARY_WINDOW,
    PAD_AFTER,
    PAD_BEFORE,
    SUBWIN_OVERLAP,
    TARGET_SR,
)


@dataclass
class AudioChunk:
    """Description of an extracted audio chunk."""

    segment_index: int
    start: float
    end: float
    path: Path


@dataclass
class Segment:
    """Simple diarization segment container."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class TemporaryAudioDirectory:
    """Context manager that cleans up extracted temporary files."""

    def __init__(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp(prefix="chunks_"))

    @property
    def path(self) -> Path:
        return self._tmpdir

    def cleanup(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __enter__(self) -> "TemporaryAudioDirectory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.cleanup()


def have_ffmpeg() -> bool:
    """Check ffmpeg availability."""
    return shutil.which("ffmpeg") is not None


def load_audio(path: Path, target_sr: int = TARGET_SR) -> Dict[str, torch.Tensor]:
    """Load audio and resample if required."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.T
    if sr != target_sr:
        channels = [soxr.resample(audio[idx], sr, target_sr) for idx in range(audio.shape[0])]
        audio = np.stack(channels, axis=0)
        sr = target_sr
    if audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True)
    waveform = torch.from_numpy(np.ascontiguousarray(audio))
    return {"waveform": waveform, "sample_rate": sr}


def audio_duration(path: Path) -> float:
    try:
        info = sf.info(path)
    except Exception:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def extract_segment(
    src: Path,
    start: float,
    end: float,
    dst: Path,
    sr: int = TARGET_SR,
    pad_before: float = PAD_BEFORE,
    pad_after: float = PAD_AFTER,
) -> None:
    if not have_ffmpeg():
        raise RuntimeError("ffmpeg is required but not available on PATH.")
    duration = audio_duration(src)
    start = clamp(start - pad_before, 0.0, duration if duration > 0 else start)
    end = clamp(end + pad_after, 0.0, duration if duration > 0 else end)
    length = max(0.001, end - start)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{length:.3f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-y",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def sliding_spans(start: float, end: float) -> List[Tuple[float, float]]:
    if end - start <= MAX_CANARY_WINDOW + 1e-6:
        return [(start, end)]
    spans: List[Tuple[float, float]] = []
    cursor = start
    while cursor < end - 1e-6:
        span_end = min(end, cursor + MAX_CANARY_WINDOW)
        spans.append((cursor, span_end))
        if span_end >= end:
            break
        cursor = span_end - SUBWIN_OVERLAP
    return spans


def build_audio_chunks(path: Path, segments: Iterable[Segment]) -> Tuple[List[AudioChunk], TemporaryAudioDirectory]:
    temp_dir = TemporaryAudioDirectory()
    chunks: List[AudioChunk] = []
    try:
        for index, segment in enumerate(segments):
            for window_start, window_end in sliding_spans(segment.start, segment.end):
                filename = f"seg{index:04d}_{window_start:.2f}_{window_end:.2f}.wav"
                dst = temp_dir.path / filename
                extract_segment(path, window_start, window_end, dst)
                chunks.append(
                    AudioChunk(
                        segment_index=index,
                        start=window_start,
                        end=window_end,
                        path=dst,
                    )
                )
    except Exception:
        temp_dir.cleanup()
        raise
    return chunks, temp_dir


def seconds_to_timestamp(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    sec_int = int(seconds)
    hours = sec_int // 3600
    minutes = (sec_int % 3600) // 60
    secs = sec_int % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

