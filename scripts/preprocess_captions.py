# scripts/preprocess_captions.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from tqdm import tqdm

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.tokenizer import build_vocab_from_captions, save_vocab


def load_coco_captions(captions_json: Path) -> List[str]:
    payload = json.loads(captions_json.read_text(encoding="utf-8"))
    captions = [ann["caption"] for ann in payload["annotations"]]
    return captions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-json", type=str, default="data/raw/coco2017/annotations/captions_train2017.json")
    ap.add_argument("--out-vocab", type=str, default="data/processed/vocab.json")
    ap.add_argument("--vocab-size", type=int, default=10000)
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()

    captions_json = Path(args.captions_json)
    out_vocab = Path(args.out_vocab)

    captions = load_coco_captions(captions_json)
    vocab = build_vocab_from_captions(captions, vocab_size=args.vocab_size, min_freq=args.min_freq)
    save_vocab(vocab, out_vocab)

    print(f"Saved vocab to {out_vocab}")
    print(f"Vocab size: {len(vocab.idx2word)}")
    print("Special token ids:",
          {"pad": vocab.pad_id, "bos": vocab.bos_id, "eos": vocab.eos_id, "unk": vocab.unk_id})


if __name__ == "__main__":
    main()
