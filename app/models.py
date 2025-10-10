"""Model management (download and load)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download, login, snapshot_download
from nemo.collections.asr.models import EncDecMultiTaskModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BF16_AVAILABLE, CANARY_FILENAME, CANARY_REPO, DEVICE, LFM_REPO, MAX_NEW_TOKENS, MODELS_DIR
from .logging_utils import setup_logging

LOGGER = setup_logging()

_CANARY_MODEL: Optional[EncDecMultiTaskModel] = None
_LFM_MODEL: Optional[AutoModelForCausalLM] = None
_LFM_TOKENIZER = None
_BAD_WORDS_IDS: Optional[List[List[int]]] = None


def _ensure_hf_login() -> None:
    token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HUGGINGFACE_TOKEN environment variable must be provided.")
    login(token=token, add_to_git_credential=False)


def ensure_models() -> Dict[str, str]:
    """Download required models lazily."""
    _ensure_hf_login()
    status: Dict[str, str] = {}

    lfm_dir = MODELS_DIR / "lfm"
    lfm_dir.mkdir(parents=True, exist_ok=True)
    if not any(lfm_dir.glob("*.safetensors")):
        LOGGER.info("Downloading LFM model…")
        snapshot_download(LFM_REPO, local_dir=lfm_dir, local_dir_use_symlinks=False)
    status["lfm"] = "ok"

    canary_dir = MODELS_DIR / "canary"
    canary_dir.mkdir(parents=True, exist_ok=True)
    canary_file = canary_dir / CANARY_FILENAME
    if not canary_file.exists():
        LOGGER.info("Downloading Canary model…")
        hf_hub_download(CANARY_REPO, filename=CANARY_FILENAME, local_dir=canary_dir, local_dir_use_symlinks=False)
    status["canary"] = "ok"

    return status


def load_canary() -> EncDecMultiTaskModel:
    global _CANARY_MODEL
    if _CANARY_MODEL is not None:
        return _CANARY_MODEL
    ensure_models()
    canary_dir = MODELS_DIR / "canary"
    nemo_path = canary_dir / CANARY_FILENAME
    LOGGER.info("Loading Canary model (%s)…", nemo_path)
    model = EncDecMultiTaskModel.restore_from(str(nemo_path), map_location=DEVICE)
    try:
        decoding_cfg = model.cfg.decoding
        decoding_cfg.beam.beam_size = 1
        model.change_decoding_strategy(decoding_cfg)
    except Exception:
        pass
    for ds_name in ("train_ds", "validation_ds", "test_ds"):
        dataset_cfg = getattr(model.cfg, ds_name, None)
        if dataset_cfg is None:
            continue
        for attr in ("num_workers", "pin_memory"):
            if hasattr(dataset_cfg, attr):
                setattr(dataset_cfg, attr, 0 if attr == "num_workers" else False)
    _CANARY_MODEL = model.to(DEVICE).eval()
    return _CANARY_MODEL


def load_lfm():
    global _LFM_MODEL, _LFM_TOKENIZER, _BAD_WORDS_IDS
    if _LFM_MODEL is not None and _LFM_TOKENIZER is not None:
        return _LFM_TOKENIZER, _LFM_MODEL, _BAD_WORDS_IDS
    ensure_models()
    lfm_dir = MODELS_DIR / "lfm"
    LOGGER.info("Loading LFM model from %s", lfm_dir)
    tokenizer = AutoTokenizer.from_pretrained(lfm_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if BF16_AVAILABLE else torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(lfm_dir, local_files_only=True, torch_dtype=dtype).to(DEVICE)
    model.eval()
    bad_phrases = [
        "German:",
        "Deutsch:",
        "English:",
        "Translate",
        "Translation",
        "Output only",
        "labels",
        "explanations",
        "Note:",
        "Hinweis",
        "Anmerkung",
        "system",
        "user",
        "assistant",
    ]
    _BAD_WORDS_IDS = [tokenizer(phrase, add_special_tokens=False).input_ids for phrase in bad_phrases]
    _LFM_MODEL = model
    _LFM_TOKENIZER = tokenizer
    return tokenizer, model, _BAD_WORDS_IDS


def lfm_generate(prompts: List[str], max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
    tokenizer, model, bad_words_ids = load_lfm()
    encodings = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generations = model.generate(
            **encodings,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids,
            do_sample=False,
            top_p=1.0,
            repetition_penalty=1.02,
            no_repeat_ngram_size=3,
        )
    outputs: List[str] = []
    attn = encodings["attention_mask"]
    for batch_index in range(generations.size(0)):
        prompt_length = int(attn[batch_index].sum().item())
        new_tokens = generations[batch_index, prompt_length:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        outputs.append(text.strip())
    return outputs


def build_chat_prompt(messages: List[Dict[str, str]]) -> str:
    tokenizer, _, _ = load_lfm()
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    system_messages = "\n".join(message["content"] for message in messages if message["role"] == "system")
    user_messages = "\n".join(message["content"] for message in messages if message["role"] == "user")
    return f"<<SYS>> {system_messages} <</SYS>>\n{user_messages}\n"

