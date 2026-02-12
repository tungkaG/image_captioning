from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import captioning_xent_loss


@dataclass
class TrainConfig:
	epochs: int = 1
	lr: float = 1e-3
	weight_decay: float = 0.0
	grad_clip_norm: float = 1.0
	device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
	log_every: int = 20
	save_dir: str = "outputs/checkpoints"
	overfit_batches: int = 0  # if >0, repeat first N batches forever


def _resolve_device(device: str) -> torch.device:
	device = (device or "auto").lower()
	if device != "auto":
		return torch.device(device)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


class Trainer:
	def __init__(
		self,
		model: nn.Module,
		optimizer: torch.optim.Optimizer,
		cfg: TrainConfig,
		pad_id: int,
		scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
	) -> None:
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.cfg = cfg
		self.pad_id = int(pad_id)
		self.device = _resolve_device(cfg.device)
		self.model.to(self.device)

	def fit(self, train_loader: DataLoader) -> None:
		self.model.train()
		step = 0
		save_dir = Path(self.cfg.save_dir)
		save_dir.mkdir(parents=True, exist_ok=True)

		for epoch in range(int(self.cfg.epochs)):
			running = 0.0
			seen = 0

			for b_idx, batch in enumerate(train_loader):
				if self.cfg.overfit_batches and b_idx >= self.cfg.overfit_batches:
					break

				images = batch.images.to(self.device)
				input_ids = batch.input_ids.to(self.device)
				target_ids = batch.target_ids.to(self.device)

				out = self.model(images=images, input_ids=input_ids)
				loss = captioning_xent_loss(out.logits, target_ids, pad_id=self.pad_id)

				self.optimizer.zero_grad(set_to_none=True)
				loss.backward()

				if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
					nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))

				self.optimizer.step()
				if self.scheduler is not None:
					self.scheduler.step()

				step += 1
				running += float(loss.item())
				seen += 1

				if step % int(self.cfg.log_every) == 0:
					print(f"epoch={epoch} step={step} loss={running/max(1, seen):.4f} device={self.device.type}")
					running = 0.0
					seen = 0

			ckpt_path = save_dir / f"captioner_epoch{epoch}.pt"
			torch.save({"model": self.model.state_dict(), "epoch": epoch}, ckpt_path)
			print(f"[OK] saved checkpoint: {ckpt_path}")
