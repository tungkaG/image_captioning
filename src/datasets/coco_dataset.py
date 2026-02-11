# src/datasets/coco_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .tokenizer import Vocab, load_vocab


@dataclass(frozen=True)
class CocoPaths:
    images_dir: Path           # e.g. data/raw/train2017
    captions_json: Path        # e.g. data/raw/annotations/captions_train2017.json
    vocab_json: Path           # e.g. data/processed/vocab.json


def default_image_transform(image_size: int = 224) -> T.Compose:
    # ImageNet normalization
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CocoCaptionDataset(Dataset):
    """
    Each item corresponds to one (image, caption) pair.
    COCO has multiple captions per image; we expand them as separate samples.
    """

    def __init__(
        self,
        paths: CocoPaths,
        split: str = "train",
        max_len: int = 30,
        transform: Optional[T.Compose] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.paths = paths
        self.split = split
        self.max_len = max_len
        self.transform = transform or default_image_transform()

        self.vocab: Vocab = load_vocab(paths.vocab_json)

        payload = json.loads(Path(paths.captions_json).read_text(encoding="utf-8"))

        # images: [{id, file_name, ...}]
        # annotations: [{image_id, caption, ...}]
        images_by_id = {img["id"]: img["file_name"] for img in payload["images"]}

        samples: List[Tuple[Path, str]] = []
        for ann in payload["annotations"]:
            img_id = ann["image_id"]
            cap = ann["caption"]
            file_name = images_by_id.get(img_id)
            if file_name is None:
                continue
            img_path = Path(paths.images_dir) / file_name
            samples.append((img_path, cap))

        if limit is not None:
            samples = samples[:limit]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, caption = self.samples[idx]

        # Load image
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            image_tensor = self.transform(img)

        # Encode caption
        # We create input_ids (BOS ...), and target_ids (... EOS) by shifting
        # This is the standard teacher-forcing format.
        full_ids = self.vocab.encode(caption, max_len=self.max_len, add_bos=True, add_eos=True)

        # Ensure at least BOS + EOS
        if len(full_ids) < 2:
            full_ids = [self.vocab.bos_id, self.vocab.eos_id]

        # Shift
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(full_ids[-1:], dtype=torch.long) if len(full_ids) == 1 else torch.tensor(full_ids[1:], dtype=torch.long)

        # Length is number of non-pad tokens in input sequence
        length = int(input_ids.numel())

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "length": length,
            "pad_id": self.vocab.pad_id,
            "caption_raw": caption,
            "image_path": str(img_path),
        }
