# src/datasets/collate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch


@dataclass
class Batch:
    images: torch.Tensor          # [B, 3, H, W]
    input_ids: torch.Tensor       # [B, T]
    target_ids: torch.Tensor      # [B, T]
    attention_mask: torch.Tensor  # [B, T] 1 for tokens, 0 for pad
    lengths: torch.Tensor         # [B]


def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads variable-length 1D Long tensors to [B, T].
    Returns padded tensor and attention mask.
    """
    lengths = torch.tensor([s.numel() for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) else 0

    padded = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)

    for i, s in enumerate(seqs):
        L = s.numel()
        padded[i, :L] = s
        mask[i, :L] = 1

    return padded, mask


def coco_collate_fn(batch: List[Dict[str, Any]]) -> Batch:
    images = torch.stack([x["image"] for x in batch], dim=0)

    input_seqs = [x["input_ids"] for x in batch]
    target_seqs = [x["target_ids"] for x in batch]
    pad_id = int(batch[0]["pad_id"])

    input_ids, attention_mask = pad_1d(input_seqs, pad_value=pad_id)
    target_ids, _ = pad_1d(target_seqs, pad_value=pad_id)
    lengths = torch.tensor([int(x["length"]) for x in batch], dtype=torch.long)

    return Batch(
        images=images,
        input_ids=input_ids,
        target_ids=target_ids,
        attention_mask=attention_mask,
        lengths=lengths,
    )
