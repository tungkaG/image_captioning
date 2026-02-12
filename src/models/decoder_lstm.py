from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DecoderOutput:
	logits: torch.Tensor  # [B, T, V]


class LSTMDecoder(nn.Module):
	"""A simple LSTM decoder conditioned on image features.

	Standard baseline:
	- embed tokens
	- prepend a projected image feature as the first time step
	- run LSTM
	- project to vocab logits
	"""

	def __init__(
		self,
		vocab_size: int,
		encoder_dim: int,
		embed_dim: int = 256,
		hidden_dim: int = 512,
		num_layers: int = 1,
		dropout: float = 0.1,
		pad_id: int = 0,
	) -> None:
		super().__init__()
		self.vocab_size = int(vocab_size)
		self.pad_id = int(pad_id)

		self.token_embed = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.pad_id)
		self.img_proj = nn.Linear(encoder_dim, embed_dim)

		self.lstm = nn.LSTM(
			input_size=embed_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		self.dropout = nn.Dropout(dropout)
		self.vocab_head = nn.Linear(hidden_dim, self.vocab_size)

	def forward(
		self,
		encoder_features: torch.Tensor,
		input_ids: torch.Tensor,
		hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
	) -> DecoderOutput:
		"""encoder_features: [B, D], input_ids: [B, T]."""
		tok = self.token_embed(input_ids)              # [B, T, E]
		img = self.img_proj(encoder_features).unsqueeze(1)  # [B, 1, E]

		# Condition by adding image as step 0. Model outputs T+1 steps; we drop the first.
		x = torch.cat([img, tok], dim=1)               # [B, 1+T, E]
		out, hidden_out = self.lstm(x, hidden)         # [B, 1+T, H]
		out = out[:, 1:, :]                            # [B, T, H]
		out = self.dropout(out)
		logits = self.vocab_head(out)                  # [B, T, V]
		return DecoderOutput(logits=logits)
