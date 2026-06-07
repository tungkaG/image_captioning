from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def beam_search_decode(
	model,
	image_tensor: torch.Tensor,
	start_token_id: int,
	eos_token_id: Optional[int] = None,
	pad_token_id: Optional[int] = None,
	max_len: int = 30,
	beam_size: int = 3,
	top_k: int = 3,
	length_penalty_alpha: float = 0.7,
) -> List[Tuple[List[int], float]]:
	"""Beam search for a single image.

	Returns top-k (token_ids, normalized_score), best first.
	"""
	model.eval()
	device = image_tensor.device
	beam_size = max(1, int(beam_size))
	top_k = max(1, int(top_k))

	beams: List[Tuple[List[int], float, bool]] = [([int(start_token_id)], 0.0, False)]

	for _ in range(max_len):
		candidates: List[Tuple[List[int], float, bool]] = []
		all_finished = True

		for tokens, score, finished in beams:
			if finished:
				candidates.append((tokens, score, True))
				continue

			all_finished = False
			input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
			out = model(images=image_tensor, input_ids=input_ids)
			log_probs = F.log_softmax(out.logits[:, -1, :], dim=-1).squeeze(0)

			if pad_token_id is not None:
				log_probs[int(pad_token_id)] = -1e9

			top_vals, top_idx = torch.topk(log_probs, k=beam_size)
			for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
				next_tokens = tokens + [int(idx)]
				next_score = float(score + val)
				done = eos_token_id is not None and int(idx) == int(eos_token_id)
				candidates.append((next_tokens, next_score, done))

		def _rank_key(item: Tuple[List[int], float, bool]) -> float:
			toks, sc, _ = item
			length = max(1, len(toks) - 1)
			norm = sc / (length ** float(length_penalty_alpha))
			return norm

		candidates.sort(key=_rank_key, reverse=True)
		beams = candidates[:beam_size]

		if all_finished:
			break

	final: List[Tuple[List[int], float]] = []
	for toks, sc, _ in beams:
		length = max(1, len(toks) - 1)
		norm = sc / (length ** float(length_penalty_alpha))
		final.append((toks, float(norm)))

	final.sort(key=lambda x: x[1], reverse=True)
	return final[:top_k]
