# scripts/preprocess_captions.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.tokenizer import build_hf_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="HF tokenizer name (e.g. gpt2, distilgpt2, bert-base-uncased).",
    )
    ap.add_argument(
        "--out-tokenizer",
        type=str,
        default="data/processed/tokenizer",
        help="Output directory for tokenizer.save_pretrained().",
    )
    ap.add_argument("--max-len", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out_tokenizer)

    tok = build_hf_tokenizer(model_name=args.model_name, max_len=args.max_len)
    tok.save(out_dir)

    print(f"Saved HF tokenizer '{args.model_name}' to {out_dir}")
    print("Tokenizer special IDs:",
          {"pad": tok.pad_id, "bos": tok.bos_id, "eos": tok.eos_id})


if __name__ == "__main__":
    main()
