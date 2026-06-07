from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.tokenizer import CaptionTokenizer
from src.inference.predict import build_model_from_checkpoint, load_image_tensor, predict_caption


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Predict caption for one image from COCO and compare with GT captions.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file.")
    ap.add_argument("--tokenizer-dir", type=str, default="data/processed/tokenizer")
    ap.add_argument("--images-dir", type=str, default="data/raw/coco2017/val2017")
    ap.add_argument(
        "--captions-json",
        type=str,
        default="data/raw/coco2017/annotations/captions_val2017.json",
    )
    ap.add_argument("--image-id", type=int, default=None, help="Specific COCO image id.")
    ap.add_argument("--index", type=int, default=None, help="Index into sorted unique image list.")
    ap.add_argument("--random", action="store_true", help="Pick a random image if image-id/index not set.")
    ap.add_argument("--strategy", type=str, default="beam", choices=["greedy", "beam"])
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", type=str, default="auto")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    payload = json.loads(Path(args.captions_json).read_text(encoding="utf-8"))
    id_to_name = {int(x["id"]): str(x["file_name"]) for x in payload["images"]}
    refs_by_image = {}
    for ann in payload["annotations"]:
        refs_by_image.setdefault(int(ann["image_id"]), []).append(str(ann["caption"]))

    image_ids = sorted([x for x in refs_by_image.keys() if x in id_to_name])
    if not image_ids:
        raise RuntimeError("No image/caption pairs found in captions_json.")

    if args.image_id is not None:
        image_id = int(args.image_id)
    elif args.index is not None:
        i = max(0, min(int(args.index), len(image_ids) - 1))
        image_id = int(image_ids[i])
    elif args.random:
        image_id = int(random.choice(image_ids))
    else:
        image_id = int(image_ids[0])

    file_name = id_to_name[image_id]
    refs = refs_by_image[image_id]
    image_path = Path(args.images_dir) / file_name

    tokenizer = CaptionTokenizer.load(Path(args.tokenizer_dir), max_len=int(args.max_len))
    model, _, dev = build_model_from_checkpoint(
        checkpoint_path=Path(args.ckpt),
        tokenizer=tokenizer,
        device=str(args.device),
    )

    image_tensor = load_image_tensor(image_path=image_path, image_size=int(args.image_size)).to(dev)
    result = predict_caption(
        model=model,
        tokenizer=tokenizer,
        image_tensor=image_tensor,
        strategy=str(args.strategy),
        beam_size=int(args.beam_size),
        top_k=int(args.top_k),
        max_len=int(args.max_len),
    )

    print(f"image_id: {image_id}")
    print(f"file_name: {file_name}")
    print(f"image_path: {image_path}")
    print(f"strategy: {result.strategy}")
    print(f"prediction: {result.caption}")

    print("\nground_truth_captions:")
    for ref in refs[:5]:
        print(f"  - {ref}")

    if result.beams:
        print("\nbeams:")
        for i, b in enumerate(result.beams):
            print(f"  [{i}] score={b['score']:.4f} caption={b['caption']}")


if __name__ == "__main__":
    main()
