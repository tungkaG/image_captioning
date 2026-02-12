from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .encoder_resnet import ResNetEncoder
from .decoder_lstm import LSTMDecoder


@dataclass
class CaptionerOutput:
	logits: torch.Tensor  # [B, T, V]


class Captioner(nn.Module):
	"""Convenience wrapper for (encoder -> decoder) captioning models."""

	def __init__(self, encoder: ResNetEncoder, decoder: LSTMDecoder) -> None:
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> CaptionerOutput:
		enc = self.encoder(images)
		dec = self.decoder(enc.features, input_ids)
		return CaptionerOutput(logits=dec.logits)

	@torch.no_grad()
	def encode_images(self, images: torch.Tensor) -> torch.Tensor:
		return self.encoder(images).features
