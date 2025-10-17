"""Helpers to interact with external inference APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence

import requests

from .config import API_BASE_URL, API_KEY, CANARY_API_MODEL, LLM_MODEL, MAX_NEW_TOKENS
from .logging_utils import setup_logging

LOGGER = setup_logging()
_SESSION = requests.Session()


def _build_headers(extra: Optional[MutableMapping[str, str]] = None) -> MutableMapping[str, str]:
    headers: MutableMapping[str, str] = {"Accept": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    if extra:
        headers.update(extra)
    return headers


def _post(
    path: str,
    *,
    json: Optional[dict] = None,
    data: Optional[dict] = None,
    files: Optional[dict] = None,
    headers: Optional[MutableMapping[str, str]] = None,
    timeout: int = 600,
):
    url = f"{API_BASE_URL}/{path.lstrip('/')}"
    merged_headers = _build_headers(headers)
    response = _SESSION.post(url, json=json, data=data, files=files, headers=merged_headers, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - informative logging
        LOGGER.error("API request to %s failed: %s", url, exc)
        LOGGER.debug("Response content: %s", response.text)
        raise
    if "application/json" in response.headers.get("Content-Type", ""):
        return response.json()
    return response.text


def ensure_models() -> Dict[str, str]:
    """Return a static status map indicating API-backed models."""
    return {"canary": CANARY_API_MODEL, "llm": LLM_MODEL}


def canary_transcribe_api(paths: Iterable[str], *, language: str = "fr", translate: bool = True) -> List[str]:
    """Transcribe audio files via the Canary API endpoint."""
    outputs: List[str] = []
    for path in paths:
        with open(path, "rb") as audio_file:
            file_name = Path(path).name
            files = {"file": (file_name, audio_file, "application/octet-stream")}
            data = {
                "model": CANARY_API_MODEL,
                "response_format": "json",
                "temperature": "0",
                "translate": "true" if translate else "false",
            }
            if language:
                data["language"] = language
            response = _post("audio/transcriptions", data=data, files=files)
        if isinstance(response, dict):
            text = (
                response.get("text")
                or response.get("translation_text")
                or response.get("transcription")
                or response.get("result")
                or ""
            )
        else:
            text = str(response)
        outputs.append(text.strip())
    return outputs


def chat_generate(messages_list: Sequence[Sequence[Dict[str, str]]], *, max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
    """Generate chat completions using the remote LLM."""
    results: List[str] = []
    for messages in messages_list:
        payload = {
            "model": LLM_MODEL,
            "messages": list(messages),
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        response = _post("chat/completions", json=payload, timeout=300)
        if isinstance(response, dict) and response.get("choices"):
            content = response["choices"][0]["message"].get("content", "")
        else:
            content = str(response)
        results.append((content or "").strip())
    return results


__all__ = ["canary_transcribe_api", "chat_generate", "ensure_models"]
