from __future__ import annotations

from typing import List, Optional

import torch


@torch.no_grad()
def greedy_decode(
	model,
	image_tensor: torch.Tensor,
	start_token_id: int,
	eos_token_id: Optional[int] = None,
	max_len: int = 30,
) -> List[int]:
	"""Greedy autoregressive decoding for a single image.

	image_tensor is expected to be [1, 3, H, W].
	Returned sequence includes the start token and generated tokens.
	"""
	model.eval()

	device = image_tensor.device
	generated = [int(start_token_id)]

	for _ in range(max_len):
		input_ids = torch.tensor([generated], dtype=torch.long, device=device)
		out = model(images=image_tensor, input_ids=input_ids)
		next_token_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
		generated.append(next_token_id)

		if eos_token_id is not None and next_token_id == int(eos_token_id):
			break

	return generated


def strip_special_tokens(
	token_ids: List[int],
	start_token_id: Optional[int],
	eos_token_id: Optional[int],
	pad_token_id: Optional[int],
) -> List[int]:
	out: List[int] = []
	for i, tid in enumerate(token_ids):
		if i == 0 and start_token_id is not None and tid == int(start_token_id):
			continue
		if eos_token_id is not None and tid == int(eos_token_id):
			break
		if pad_token_id is not None and tid == int(pad_token_id):
			continue
		out.append(int(tid))
	return out
