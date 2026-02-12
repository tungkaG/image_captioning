from __future__ import annotations

import torch
import torch.nn as nn


def captioning_xent_loss(
	logits: torch.Tensor,
	target_ids: torch.Tensor,
	pad_id: int,
) -> torch.Tensor:
	"""Token-level cross entropy, ignoring PAD.

	logits: [B, T, V]
	target_ids: [B, T]
	"""
	b, t, v = logits.shape
	logits_2d = logits.reshape(b * t, v)
	targets_1d = target_ids.reshape(b * t)
	return nn.functional.cross_entropy(logits_2d, targets_1d, ignore_index=int(pad_id))
