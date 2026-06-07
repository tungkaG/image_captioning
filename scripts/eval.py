from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.evaluate import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a trained ResNet+LSTM captioner on COCO captions.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file.")
    ap.add_argument("--tokenizer-dir", type=str, default="data/processed/tokenizer")
    ap.add_argument("--images-dir", type=str, default="data/raw/coco2017/val2017")
    ap.add_argument(
        "--captions-json",
        type=str,
        default="data/raw/coco2017/annotations/captions_val2017.json",
    )
    ap.add_argument("--strategy", type=str, default="beam", choices=["greedy", "beam"])
    ap.add_argument("--beam-size", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--show-samples", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    metrics, samples = evaluate_checkpoint(
        checkpoint_path=Path(args.ckpt),
        tokenizer_dir=Path(args.tokenizer_dir),
        images_dir=Path(args.images_dir),
        captions_json=Path(args.captions_json),
        strategy=args.strategy,
        beam_size=int(args.beam_size),
        max_len=int(args.max_len),
        limit=int(args.limit) if args.limit and args.limit > 0 else None,
        image_size=int(args.image_size),
        device=str(args.device),
    )

    print("\n=== Metrics ===")
    print(f"images_evaluated: {metrics['num_images']}")
    print(f"bleu1: {metrics['bleu1']:.4f}")
    print(f"bleu2: {metrics['bleu2']:.4f}")
    print(f"bleu3: {metrics['bleu3']:.4f}")
    print(f"bleu4: {metrics['bleu4']:.4f}")
    print(f"avg_pred_len_tokens: {metrics['pred_len_avg']:.2f}")

    show_n = max(0, int(args.show_samples))
    if show_n:
        print("\n=== Qualitative samples ===")
        for idx, item in enumerate(samples[:show_n]):
            print(f"\n[{idx}] image_id={item.image_id} file={item.file_name}")
            print(f"pred: {item.prediction}")
            print("gt:")
            for ref in item.references[:5]:
                print(f"  - {ref}")


if __name__ == "__main__":
    main()
