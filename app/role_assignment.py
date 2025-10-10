"""Speaker aliasing using the LFM model without predefined titles."""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .models import build_chat_prompt, lfm_generate

_ARRAY_RE = re.compile(r"\[[\s\S]*\]")
_WORD_RE = re.compile(r"[\w'\-]+", re.UNICODE)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _aggregate_turns(turns: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    aggregated: Dict[str, List[str]] = {}
    for speaker, text in turns:
        aggregated.setdefault(speaker, []).append(_normalize(text))
    return {speaker: _normalize(" ".join(parts)) for speaker, parts in aggregated.items()}


def _format_conversation(turns: Iterable[Tuple[str, str]]) -> str:
    lines = []
    for speaker, text in turns:
        text = _normalize(text)
        if not text:
            continue
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _alias_tokens(alias: str) -> List[str]:
    return [token.lower() for token in _WORD_RE.findall(alias) if token]


def _evidence_contains_alias(alias: str, evidence: str) -> bool:
    alias_tokens = _alias_tokens(alias)
    if not alias_tokens:
        return False
    evidence_tokens = [token.lower() for token in _WORD_RE.findall(evidence)]
    evidence_set = set(evidence_tokens)
    return all(token in evidence_set for token in alias_tokens)


def _validate_entry(entry: dict, per_speaker_text: Dict[str, str], conversation_text: str) -> Optional[str]:
    speaker = entry.get("speaker")
    alias = _normalize(entry.get("alias", ""))
    evidence = _normalize(entry.get("evidence", ""))
    if not speaker or speaker not in per_speaker_text:
        return None
    if not alias:
        return None
    if not evidence:
        return None
    if evidence.lower() not in conversation_text.lower():
        return None
    if not _evidence_contains_alias(alias, evidence):
        return None
    return alias


def request_aliases(turns: List[Tuple[str, str]]) -> Dict[str, str]:
    if not turns:
        return {}
    per_speaker_text = _aggregate_turns(turns)
    conversation = _format_conversation(turns)
    instructions = (
        "You analyse conversation transcripts. "
        "Assign one short alias (name or role) to each speaker ID. "
        "Use only names or roles that appear verbatim in the transcript. "
        "If a name is not stated, you may use a role mentioned for that speaker. "
        "Never invent facts and never output placeholders like 'Speaker'. "
        "Return JSON array with objects: {\"speaker\": str, \"alias\": str, \"evidence\": str, \"confidence\": float}. "
        "Evidence must be an exact quotation from the transcript that justifies the alias." 
    )
    prompt = (
        f"Conversation:\n{conversation}\n\n"
        "Provide aliases now."
    )
    chat_prompt = build_chat_prompt(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ]
    )
    response = lfm_generate([chat_prompt])[0]
    match = _ARRAY_RE.search(response)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    aliases: Dict[str, str] = {}
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict):
            continue
        alias = _validate_entry(entry, per_speaker_text, conversation)
        if alias:
            aliases[entry["speaker"]] = alias
    return aliases

