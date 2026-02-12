from __future__ import annotations

import torch


def build_optimizer(params, lr: float = 1e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
	return torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
