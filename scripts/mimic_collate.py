"""
scripts/mimic_collate.py

Now loads 1 or 2 real images from:
    data/raw/coco2017/train2017

If directory is missing or empty, falls back to synthetic tensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.collate import coco_collate_fn, pad_1d
from src.datasets.coco_dataset import default_image_transform


COCO_TRAIN_DIR = ROOT / "data/raw/coco2017/train2017"


def _load_real_images(num_images: int) -> list[torch.Tensor]:
    """
    Load 1 or 2 real images from COCO train2017 directory.
    Returns list of tensors [3, H, W].
    """
    if not COCO_TRAIN_DIR.exists():
        print(f"[WARN] {COCO_TRAIN_DIR} not found. Falling back to synthetic tensors.")
        return []

    image_files = list(COCO_TRAIN_DIR.glob("*.jpg"))
    if len(image_files) == 0:
        print(f"[WARN] No .jpg images found in {COCO_TRAIN_DIR}.")
        return []

    random.shuffle(image_files)
    image_files = image_files[:num_images]

    tfm = default_image_transform(image_size=224)
    tensors = []

    for p in image_files:
        with Image.open(p) as img:
            img = img.convert("RGB")
            tensor = tfm(img)
            tensors.append(tensor)
            print(f"[INFO] Loaded real image: {p.name}")

    return tensors


def _synthetic_image() -> torch.Tensor:
    """Fallback synthetic image tensor [3, 2, 2]."""
    return torch.arange(1, 13, dtype=torch.float32).reshape(3, 2, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num-images",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of real images to load from COCO train2017.",
    )
    args = ap.parse_args()

    pad_id = 0

    real_images = _load_real_images(args.num_images)

    if len(real_images) == 0:
        # fallback to synthetic
        real_images = [_synthetic_image() for _ in range(args.num_images)]

    if len(real_images) == 1:
        real_images = [real_images[0], real_images[0].clone()]

    img_a, img_b = real_images[:2]

    sample_a = {
        "image": img_a,
        "input_ids": torch.tensor([1, 10, 57, 2], dtype=torch.long),
        "target_ids": torch.tensor([10, 57, 2], dtype=torch.long),
        "length": 4,
        "pad_id": pad_id,
        "caption_raw": "a dog",
        "image_path": "real/a.jpg",
    }

    sample_b = {
        "image": img_b,
        "input_ids": torch.tensor([1, 10, 57, 214, 2], dtype=torch.long),
        "target_ids": torch.tensor([10, 57, 214, 2], dtype=torch.long),
        "length": 5,
        "pad_id": pad_id,
        "caption_raw": "a dog runs",
        "image_path": "real/b.jpg",
    }

    batch_list = [sample_a, sample_b]

    print("### What DataLoader passes into collate_fn")
    for i, s in enumerate(batch_list):
        print(f"\nsample {i} keys:", sorted(s.keys()))
        print(f"  image shape: {tuple(s['image'].shape)}")
        print(f"  image dtype: {s['image'].dtype}")
        print(f"  image min/max: {float(s['image'].min()):.4f} / {float(s['image'].max()):.4f}")
        print(f"  input_ids:   {s['input_ids'].tolist()}")
        print(f"  target_ids:  {s['target_ids'].tolist()}")
        print(f"  length:      {s['length']}")
        print(f"  pad_id:      {s['pad_id']}")

    print("\n### pad_1d(input_ids) output")
    padded_inp, mask = pad_1d(
        [sample_a["input_ids"], sample_b["input_ids"]],
        pad_value=pad_id,
    )
    print("padded_inp shape:", tuple(padded_inp.shape))
    print(padded_inp)
    print("attention_mask shape:", tuple(mask.shape))
    print(mask)

    print("\n### Now calling your real coco_collate_fn")
    batch = coco_collate_fn(batch_list)

    print("batch.images shape:", tuple(batch.images.shape))
    print("batch.input_ids shape:", tuple(batch.input_ids.shape))
    print(batch.input_ids)
    print("batch.target_ids shape:", tuple(batch.target_ids.shape))
    print(batch.target_ids)
    print("batch.attention_mask shape:", tuple(batch.attention_mask.shape))
    print(batch.attention_mask)
    print("batch.lengths:", batch.lengths.tolist())

    pad_positions = batch.input_ids.eq(pad_id)
    if pad_positions.any():
        assert torch.all(batch.attention_mask[pad_positions] == 0), \
            "Mask mismatch on pad positions"

    print("\nOK: padding + masks consistent.")


if __name__ == "__main__":
    main()
