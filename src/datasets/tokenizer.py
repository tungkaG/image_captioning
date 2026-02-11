# src/datasets/tokenizer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class CaptionTokenizer:
    """
    Hugging Face tokenizer wrapper for captioning.
    - Handles encode/decode
    - Returns input_ids + attention_mask
    - Supports saving/loading via save_pretrained/from_pretrained
    """
    tokenizer: PreTrainedTokenizerBase
    max_len: int = 30

    @property
    def pad_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def bos_id(self) -> Optional[int]:
        return None if self.tokenizer.bos_token_id is None else int(self.tokenizer.bos_token_id)

    @property
    def eos_id(self) -> Optional[int]:
        return None if self.tokenizer.eos_token_id is None else int(self.tokenizer.eos_token_id)

    def encode(self, caption: str, max_len: Optional[int] = None) -> Dict[str, List[int]]:
        """
        Returns dict with:
          - input_ids: List[int]
          - attention_mask: List[int]
        """
        max_len = max_len or self.max_len
        out = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    def batch_encode(self, captions: List[str], max_len: Optional[int] = None) -> Dict[str, List[List[int]]]:
        max_len = max_len or self.max_len
        out = self.tokenizer(
            captions,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).strip()

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(path))

    @classmethod
    def load(cls, path: Union[str, Path], max_len: int = 30) -> "CaptionTokenizer":
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        return cls(tokenizer=tok, max_len=max_len)


def build_hf_tokenizer(
    model_name: str = "gpt2",
    max_len: int = 30,
) -> CaptionTokenizer:
    """
    Recommended portfolio default: GPT-2 tokenizer (BPE).
    For captioning, padding is needed; GPT-2 doesn't define pad_token by default.
    We set pad_token = eos_token (common practice).
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure we have a dedicated pad token (best practice for batching + loss masking)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})

    return CaptionTokenizer(tokenizer=tok, max_len=max_len)
