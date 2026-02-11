# scripts/debug_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.coco_dataset import CocoCaptionDataset, CocoPaths
from src.datasets.collate import coco_collate_fn
from src.datasets.tokenizer import CaptionTokenizer  # <-- HF wrapper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=str, default="data/raw/coco2017/train2017")
    ap.add_argument("--captions-json", type=str, default="data/raw/coco2017/annotations/captions_train2017.json")
    ap.add_argument(
        "--tokenizer-dir",
        type=str,
        default="data/processed/tokenizer",
        help="Directory created by tokenizer.save_pretrained (HF).",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--limit", type=int, default=64)
    args = ap.parse_args()

    paths = CocoPaths(
        images_dir=Path(args.images_dir),
        captions_json=Path(args.captions_json),
        vocab_json=Path(args.tokenizer_dir),  # keep field name if your CocoPaths expects vocab_json
    )

    # Load tokenizer
    tok = CaptionTokenizer.load(Path(args.tokenizer_dir), max_len=args.max_len)
    ds = CocoCaptionDataset(paths=paths, max_len=args.max_len, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=coco_collate_fn)

    batch = next(iter(dl))
    print("images:", tuple(batch.images.shape))
    print("input_ids:", tuple(batch.input_ids.shape))
    print("target_ids:", tuple(batch.target_ids.shape))
    print("attention_mask:", tuple(batch.attention_mask.shape))
    print("lengths:", batch.lengths.tolist())

    # Decode first 2 examples
    for i in range(min(2, batch.input_ids.size(0))):
        inp = batch.input_ids[i].tolist()
        tgt = batch.target_ids[i].tolist()
        print("\nExample", i)
        print("Decoded input:", tok.decode(inp, skip_special_tokens=True))
        print("Decoded target:", tok.decode(tgt, skip_special_tokens=True))
        print("Note: target is shifted by 1 token (teacher forcing).")

    # Pad sanity: wherever attention_mask == 0, input_ids should be pad_id
    pad_id = tok.pad_id
    pad_positions = (batch.attention_mask == 0)
    if pad_positions.any():
        assert torch.all(batch.input_ids[pad_positions] == pad_id), "input_ids not PAD where mask==0"
    print("\n✅ Collate padding + masks look consistent.")



if __name__ == "__main__":
    main()
