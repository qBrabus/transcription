"""Helpers for invoking the remote language models via HTTP API."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import requests

from .config import (
    API_BASE_URL,
    API_CHAT_MODEL,
    API_KEY,
    API_TIMEOUT,
    API_TRANSCRIPTION_MODEL,
    MAX_NEW_TOKENS,
)
from .logging_utils import setup_logging

LOGGER = setup_logging()


def ensure_models() -> Dict[str, str]:
    """Validate configuration before running the pipeline."""

    if not API_CHAT_MODEL:
        raise RuntimeError(
            "TRANSCRIPTION_CHAT_MODEL environment variable must be provided when using the remote API."
        )
    if not API_TRANSCRIPTION_MODEL:
        raise RuntimeError(
            "TRANSCRIPTION_TRANSCRIPTION_MODEL environment variable must be provided when using the remote API."
        )
    LOGGER.info(
        "Using remote transcription service at %s (chat model: %s, transcription model: %s)",
        API_BASE_URL,
        API_CHAT_MODEL,
        API_TRANSCRIPTION_MODEL,
    )
    return {"chat": "ok", "transcription": "ok"}


def _build_headers(content_type: str = "application/json") -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": content_type}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def _build_url(path: str) -> str:
    base = API_BASE_URL.rstrip("/")
    endpoint = path.lstrip("/")
    return f"{base}/{endpoint}"


def _post_chat(messages: Sequence[Dict[str, str]], max_new_tokens: int) -> str:
    url = _build_url("chat/completions")
    payload = {
        "model": API_CHAT_MODEL,
        "messages": list(messages),
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
    }
    try:
        response = requests.post(url, headers=_build_headers(), json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failure path
        raise RuntimeError("Remote chat completion failed") from exc
    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        LOGGER.warning("Chat completion returned no choices: %s", data)
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    return str(content).strip()


def chat_generate_batch(
    batches: Iterable[Sequence[Dict[str, str]]], max_new_tokens: int = MAX_NEW_TOKENS
) -> List[str]:
    """Generate responses for a list of chat message batches."""

    outputs: List[str] = []
    for batch in batches:
        outputs.append(_post_chat(batch, max_new_tokens))
    return outputs

