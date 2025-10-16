"""Transcription (FR -> EN) via remote API."""

from __future__ import annotations

import os
from typing import Iterable, List

import requests

from .config import (
    API_BASE_URL,
    API_KEY,
    API_TRANSCRIPTION_MODEL,
    API_TRANSCRIPTION_TIMEOUT,
)
from .logging_utils import setup_logging

LOGGER = setup_logging()


def _build_headers() -> dict:
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _transcribe_file(path: str) -> str:
    url = f"{API_BASE_URL.rstrip('/')}/audio/transcriptions"
    data = {
        "model": API_TRANSCRIPTION_MODEL,
        "language": "fr",
        "translate": True,
        "response_format": "json",
    }
    with open(path, "rb") as handle:
        files = {"file": (os.path.basename(path), handle, "application/octet-stream")}
        try:
            LOGGER.debug("Requesting transcription for %s", path)
            response = requests.post(
                url,
                headers=_build_headers(),
                data=data,
                files=files,
                timeout=API_TRANSCRIPTION_TIMEOUT,
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            raise RuntimeError(f"Remote transcription failed for {path}") from exc
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()
    if isinstance(payload, dict):
        if isinstance(payload.get("text"), str):
            return payload["text"].strip()
        data_entries = payload.get("data")
        if isinstance(data_entries, list) and data_entries:
            entry = data_entries[0]
            if isinstance(entry, dict) and isinstance(entry.get("text"), str):
                return entry["text"].strip()
    return str(payload).strip()


def canary_transcribe(paths: Iterable[str], batch_size: int = 6) -> List[str]:
    del batch_size  # batching is handled server-side
    return [_transcribe_file(path) for path in paths]

