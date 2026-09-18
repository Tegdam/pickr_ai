"""Qwen2.5 tokenizer wrapper: renders messages through the model's own chat
template and counts tokens the way the benchmark's engines will.

Traces are exported pre-rendered so `vllm bench serve` can send raw text to
the completions endpoint with --skip-chat-template. That removes the
cross-engine chat-template parity risk (spec §4) by construction: both
engines receive byte-identical prompts.
"""
from __future__ import annotations

import hashlib


class QwenTokenizer:
    def __init__(self, hf_tokenizer, model_id: str, revision: str | None):
        self._tok = hf_tokenizer
        self.model_id = model_id
        self.revision = revision

    @classmethod
    def load(cls, model_id: str = "Qwen/Qwen2.5-3B-Instruct", revision: str | None = None) -> "QwenTokenizer":
        from huggingface_hub import model_info
        from transformers import AutoTokenizer

        # Resolve the commit first and download at exactly that commit, so the
        # revision recorded in meta.json is the one the template came from.
        resolved = revision or model_info(model_id).sha
        tok = AutoTokenizer.from_pretrained(model_id, revision=resolved)
        return cls(tok, model_id, resolved)

    def render(self, messages: list[dict]) -> str:
        return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count(self, text: str) -> int:
        return len(self._tok(text).input_ids)

    @property
    def template_sha256(self) -> str:
        template = getattr(self._tok, "chat_template", "") or ""
        return hashlib.sha256(template.encode("utf-8")).hexdigest()
