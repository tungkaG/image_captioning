from __future__ import annotations

from typing import Optional

import torch


def build_scheduler(
	optimizer: torch.optim.Optimizer,
	kind: str = "none",
	**kwargs,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
	if kind in (None, "none", ""):
		return None
	if kind == "step":
		step_size = int(kwargs.get("step_size", 1))
		gamma = float(kwargs.get("gamma", 0.1))
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
	raise ValueError(f"Unknown scheduler kind: {kind}")
