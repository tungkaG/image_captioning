from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class EncoderOutput:
	features: torch.Tensor  # [B, D]


class ResNetEncoder(nn.Module):
	"""Pretrained ResNet encoder that returns a single feature vector per image.

	This is the classic 'Show and Tell' style encoder: take a CNN pretrained on ImageNet,
	pool to a vector (the penultimate layer), then optionally project to a smaller dim.
	"""

	def __init__(
		self,
		name: str = "resnet50",
		pretrained: bool = True,
		trainable: bool = False,
		proj_dim: int = 512,
	) -> None:
		super().__init__()

		if name == "resnet18":
			backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
			in_dim = 512
		elif name == "resnet34":
			backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
			in_dim = 512
		elif name == "resnet50":
			backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
			in_dim = 2048
		else:
			raise ValueError(f"Unsupported encoder backbone: {name}")

		# Drop the classifier; keep everything up to avgpool.
		modules = list(backbone.children())[:-1]
		self.backbone = nn.Sequential(*modules)

		self.proj = nn.Linear(in_dim, proj_dim) if proj_dim and proj_dim != in_dim else nn.Identity()
		self.out_dim = proj_dim if proj_dim and proj_dim != in_dim else in_dim

		self.set_trainable(trainable)

	def set_trainable(self, trainable: bool) -> None:
		for p in self.backbone.parameters():
			p.requires_grad = bool(trainable)

	def forward(self, images: torch.Tensor) -> EncoderOutput:
		"""images: [B, 3, H, W] normalized like ImageNet."""
		x = self.backbone(images)          # [B, C, 1, 1]
		x = x.flatten(1)                   # [B, C]
		x = self.proj(x)                   # [B, D]
		return EncoderOutput(features=x)
