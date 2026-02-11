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

from .tokenizer import CaptionTokenizer  # <-- HF wrapper


@dataclass(frozen=True)
class CocoPaths:
    images_dir: Path           # e.g. data/raw/train2017
    captions_json: Path        # e.g. data/raw/annotations/captions_train2017.json
    vocab_json: Path           # NOW: directory like data/processed/tokenizer/


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

    Tokenization:
    - Uses Hugging Face tokenizer (subword/BPE/etc).
    - Creates input_ids and target_ids by shifting (teacher forcing).
    - attention_mask corresponds to input_ids (0 where pad).
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

        # Load HF tokenizer wrapper from directory (saved via save_pretrained)
        self.tok: CaptionTokenizer = CaptionTokenizer.load(paths.vocab_json, max_len=max_len)
        self.pad_id: int = self.tok.pad_id

        payload = json.loads(Path(paths.captions_json).read_text(encoding="utf-8"))

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

        # Encode caption with HF tokenizer
        enc = self.tok.encode(caption, max_len=self.max_len)
        full_ids = enc["input_ids"]           # length = max_len
        full_mask = enc["attention_mask"]     # length = max_len

        # Teacher forcing shift: (BOS ... ) -> predict next token
        # input_ids  = ids[:-1]
        # target_ids = ids[1:]
        input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(full_ids[1:], dtype=torch.long)

        # attention mask aligned with input_ids
        attention_mask = torch.tensor(full_mask[:-1], dtype=torch.long)

        # length = number of non-pad tokens in input sequence (== sum(attention_mask))
        length = int(attention_mask.sum().item())

        return {
            "image": image_tensor,
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "length": length,
            "pad_id": self.pad_id,
            "caption_raw": caption,
            "image_path": str(img_path),
        }
