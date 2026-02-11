# src/datasets/tokenizer.py
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


_BASIC_TOKEN_RE = re.compile(r"[a-z0-9]+|[^\s]", re.IGNORECASE)

# TODO: can replace with more robust tokenizer if desired (e.g. HuggingFace's tokenizers library), but this simple one is sufficient for COCO captions and keeps the vocab small.
def basic_tokenize(text: str) -> List[str]:
    """
    Simple tokenizer:
    - lowercase
    - split into alphanum chunks and punctuation tokens
    """
    text = text.lower().strip()
    return _BASIC_TOKEN_RE.findall(text)


@dataclass(frozen=True)
class Vocab:
    word2idx: Dict[str, int]
    idx2word: List[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    @property
    def pad_id(self) -> int:
        return self.word2idx[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.word2idx[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.word2idx[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.word2idx[self.unk_token]

    def encode(
        self,
        caption: str,
        max_len: int = 30,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        tokens = basic_tokenize(caption)
        ids: List[int] = []

        if add_bos:
            ids.append(self.bos_id)

        for t in tokens:
            ids.append(self.word2idx.get(t, self.unk_id))

        if add_eos:
            ids.append(self.eos_id)

        # Truncate (keep at least BOS and EOS if enabled)
        if len(ids) > max_len:
            ids = ids[:max_len]
            # Ensure last token is EOS if add_eos
            if add_eos:
                ids[-1] = self.eos_id

        return ids

    def decode(self, ids: List[int], stop_at_eos: bool = True) -> str:
        words = []
        for i in ids:
            w = self.idx2word[i] if 0 <= i < len(self.idx2word) else self.unk_token
            if w == self.bos_token or w == self.pad_token:
                continue
            if stop_at_eos and w == self.eos_token:
                break
            words.append(w)
        # light detokenization (keeps punctuation separated but readable)
        return " ".join(words).replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")


def build_vocab_from_captions(
    captions: List[str],
    vocab_size: int = 10000,
    min_freq: int = 2,
) -> Vocab:
    special = ["<pad>", "<bos>", "<eos>", "<unk>"]
    counter: Counter[str] = Counter()

    for cap in captions:
        counter.update(basic_tokenize(cap)) 

    # Filter by min_freq
    words = [w for w, f in counter.items() if f >= min_freq]
    # Sort by frequency then alphabetically for determinism
    words_sorted = sorted(words, key=lambda w: (-counter[w], w))
    words_kept = words_sorted[: max(0, vocab_size - len(special))]

    idx2word = special + words_kept
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return Vocab(word2idx=word2idx, idx2word=idx2word)


def save_vocab(vocab: Vocab, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"idx2word": vocab.idx2word}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(path: str | Path) -> Vocab:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    idx2word = payload["idx2word"]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return Vocab(word2idx=word2idx, idx2word=idx2word)
