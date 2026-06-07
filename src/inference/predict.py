from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.datasets.coco_dataset import default_image_transform
from src.datasets.tokenizer import CaptionTokenizer
from src.inference.beam_search import beam_search_decode
from src.inference.greedy import greedy_decode, strip_special_tokens
from src.models.captioner import Captioner
from src.models.decoder_lstm import LSTMDecoder
from src.models.encoder_resnet import ResNetEncoder


@dataclass
class PredictionResult:
	caption: str
	token_ids: List[int]
	strategy: str
	beams: Optional[List[Dict[str, Any]]] = None


def resolve_device(device: str = "auto") -> torch.device:
	d = (device or "auto").lower()
	if d != "auto":
		return torch.device(d)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def resolve_start_token(tok: CaptionTokenizer) -> int:
	if tok.bos_id is not None:
		return int(tok.bos_id)
	if tok.eos_id is not None:
		return int(tok.eos_id)
	cls_id = getattr(tok.tokenizer, "cls_token_id", None)
	if cls_id is not None:
		return int(cls_id)
	return int(tok.pad_id)


def build_model_from_checkpoint(
	checkpoint_path: Path,
	tokenizer: CaptionTokenizer,
	device: str = "auto",
	encoder_name: Optional[str] = None,
	proj_dim: Optional[int] = None,
	embed_dim: Optional[int] = None,
	hidden_dim: Optional[int] = None,
	num_layers: Optional[int] = None,
	dropout: Optional[float] = None,
) -> tuple[Captioner, dict, torch.device]:
	"""Build a ResNet+LSTM captioner and load weights from checkpoint.

	If checkpoint contains model_config, those values are used unless explicitly overridden.
	"""
	checkpoint_path = Path(checkpoint_path)
	dev = resolve_device(device)
	ckpt = torch.load(checkpoint_path, map_location=dev)
	model_cfg = ckpt.get("model_config", {})

	enc_cfg = model_cfg.get("encoder", {})
	dec_cfg = model_cfg.get("decoder", {})

	name = encoder_name or enc_cfg.get("name", "resnet50")
	proj = int(proj_dim if proj_dim is not None else enc_cfg.get("proj_dim", 512))
	emb = int(embed_dim if embed_dim is not None else dec_cfg.get("embed_dim", 256))
	hid = int(hidden_dim if hidden_dim is not None else dec_cfg.get("hidden_dim", 512))
	layers = int(num_layers if num_layers is not None else dec_cfg.get("num_layers", 1))
	drop = float(dropout if dropout is not None else dec_cfg.get("dropout", 0.1))

	encoder = ResNetEncoder(name=name, pretrained=False, trainable=False, proj_dim=proj)
	decoder = LSTMDecoder(
		vocab_size=len(tokenizer.tokenizer),
		encoder_dim=encoder.out_dim,
		embed_dim=emb,
		hidden_dim=hid,
		num_layers=layers,
		dropout=drop,
		pad_id=int(tokenizer.pad_id),
	)
	model = Captioner(encoder=encoder, decoder=decoder).to(dev)
	model.load_state_dict(ckpt["model"])
	model.eval()
	return model, ckpt, dev


def load_image_tensor(image_path: Path, image_size: int = 224) -> torch.Tensor:
	from PIL import Image

	transform = default_image_transform(image_size=image_size)
	with Image.open(image_path) as img:
		img = img.convert("RGB")
		x = transform(img)
	return x.unsqueeze(0)


@torch.no_grad()
def predict_caption(
	model: Captioner,
	tokenizer: CaptionTokenizer,
	image_tensor: torch.Tensor,
	strategy: str = "beam",
	beam_size: int = 3,
	top_k: int = 3,
	max_len: int = 30,
) -> PredictionResult:
	start_id = resolve_start_token(tokenizer)
	eos_id = tokenizer.eos_id
	pad_id = tokenizer.pad_id

	strategy = (strategy or "beam").lower()

	if strategy == "greedy":
		ids = greedy_decode(
			model=model,
			image_tensor=image_tensor,
			start_token_id=start_id,
			eos_token_id=eos_id,
			max_len=int(max_len),
		)
		clean = strip_special_tokens(ids, start_id, eos_id, pad_id)
		caption = tokenizer.decode(clean, skip_special_tokens=True)
		return PredictionResult(caption=caption, token_ids=clean, strategy="greedy")

	beams = beam_search_decode(
		model=model,
		image_tensor=image_tensor,
		start_token_id=start_id,
		eos_token_id=eos_id,
		pad_token_id=pad_id,
		max_len=int(max_len),
		beam_size=int(beam_size),
		top_k=int(top_k),
	)

	beam_payload: List[Dict[str, Any]] = []
	for ids, score in beams:
		clean_ids = strip_special_tokens(ids, start_id, eos_id, pad_id)
		beam_payload.append(
			{
				"token_ids": clean_ids,
				"score": float(score),
				"caption": tokenizer.decode(clean_ids, skip_special_tokens=True),
			}
		)

	best = beam_payload[0]
	return PredictionResult(
		caption=str(best["caption"]),
		token_ids=list(best["token_ids"]),
		strategy="beam",
		beams=beam_payload,
	)
