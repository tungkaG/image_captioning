# src/datasets/coco_dataset.py
from __future__ import annotations

import io
import json
import random
import zipfile
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
        one_caption_per_image: bool = False,
    ) -> None:
        self.paths = paths
        self.split = split
        self.max_len = max_len
        self.transform = transform or default_image_transform()
        self.one_caption_per_image = bool(one_caption_per_image)
        self.images_root = Path(paths.images_dir)
        self.images_zip_path = self.images_root if self.images_root.suffix.lower() == ".zip" else None
        self._zip_file: Optional[zipfile.ZipFile] = None
        self._available_files: Optional[set[str]] = None
        if self.images_zip_path is None and self.images_root.exists():
            self._available_files = {path.name for path in self.images_root.iterdir() if path.is_file()}

        # Load HF tokenizer wrapper from directory (saved via save_pretrained)
        self.tok: CaptionTokenizer = CaptionTokenizer.load(paths.vocab_json, max_len=max_len)
        self.pad_id: int = self.tok.pad_id

        payload = json.loads(Path(paths.captions_json).read_text(encoding="utf-8"))

        images_by_id = {img["id"]: img["file_name"] for img in payload["images"]}

        if self.one_caption_per_image:
            grouped_samples: List[Tuple[str, List[str]]] = []
            captions_by_path: Dict[str, List[str]] = {}
            for ann in payload["annotations"]:
                img_id = ann["image_id"]
                cap = ann["caption"]
                file_name = images_by_id.get(img_id)
                if file_name is None or not self._image_exists(file_name):
                    continue
                captions_by_path.setdefault(file_name, []).append(cap)

            for image_ref, captions in captions_by_path.items():
                grouped_samples.append((image_ref, captions))

            if limit is not None:
                grouped_samples = grouped_samples[:limit]

            self.samples = grouped_samples
            return

        samples: List[Tuple[str, str]] = []
        for ann in payload["annotations"]:
            img_id = ann["image_id"]
            cap = ann["caption"]
            file_name = images_by_id.get(img_id)
            if file_name is None or not self._image_exists(file_name):
                continue
            samples.append((file_name, cap))

        if limit is not None:
            samples = samples[:limit]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_zip_file(self) -> zipfile.ZipFile:
        if self.images_zip_path is None:
            raise RuntimeError("zip access requested for non-zip dataset")
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.images_zip_path)
        return self._zip_file

    def _zip_member_name(self, image_ref: str) -> str:
        archive = self._get_zip_file()
        try:
            archive.getinfo(image_ref)
            return image_ref
        except KeyError:
            prefixed_ref = f"{self.images_zip_path.stem}/{image_ref}"
            archive.getinfo(prefixed_ref)
            return prefixed_ref

    def _image_exists(self, image_ref: str) -> bool:
        if self.images_zip_path is not None:
            try:
                self._zip_member_name(image_ref)
                return True
            except KeyError:
                return False

        if self._available_files is not None:
            return image_ref in self._available_files
        return (self.images_root / image_ref).exists()

    def _load_image(self, image_ref: str) -> Image.Image:
        if self.images_zip_path is not None:
            member_name = self._zip_member_name(image_ref)
            with self._get_zip_file().open(member_name) as handle:
                image_bytes = handle.read()
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.convert("RGB")

        with Image.open(self.images_root / image_ref) as img:
            return img.convert("RGB")

    def _display_image_path(self, image_ref: str) -> str:
        if self.images_zip_path is not None:
            return f"{self.images_zip_path}::{image_ref}"
        return str(self.images_root / image_ref)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        if self.one_caption_per_image:
            image_ref, captions = sample
            caption = random.choice(captions)
        else:
            image_ref, caption = sample

        image = self._load_image(image_ref)
        image_tensor = self.transform(image)

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
            "image_path": self._display_image_path(image_ref),
        }

    def __del__(self) -> None:
        if self._zip_file is not None:
            self._zip_file.close()
