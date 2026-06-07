from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.datasets.tokenizer import CaptionTokenizer
from src.eval.metrics import compute_bleu_scores, mean_caption_len, tokenize_caption
from src.inference.predict import build_model_from_checkpoint, load_image_tensor, predict_caption


@dataclass
class EvalItem:
	image_id: int
	file_name: str
	prediction: str
	references: List[str]


def load_coco_image_to_refs(captions_json: Path) -> List[Dict]:
	payload = json.loads(Path(captions_json).read_text(encoding="utf-8"))
	images = {int(x["id"]): str(x["file_name"]) for x in payload["images"]}
	refs_by_image: Dict[int, List[str]] = {}
	for ann in payload["annotations"]:
		img_id = int(ann["image_id"])
		refs_by_image.setdefault(img_id, []).append(str(ann["caption"]))

	items = []
	for image_id, refs in refs_by_image.items():
		file_name = images.get(image_id)
		if file_name is None:
			continue
		items.append({"image_id": image_id, "file_name": file_name, "references": refs})

	items.sort(key=lambda x: int(x["image_id"]))
	return items


def evaluate_checkpoint(
	checkpoint_path: Path,
	tokenizer_dir: Path,
	images_dir: Path,
	captions_json: Path,
	strategy: str = "beam",
	beam_size: int = 3,
	max_len: int = 30,
	limit: Optional[int] = None,
	image_size: int = 224,
	device: str = "auto",
	top_k: int = 3,
) -> tuple[dict, List[EvalItem]]:
	tokenizer = CaptionTokenizer.load(tokenizer_dir, max_len=max_len)
	model, _, dev = build_model_from_checkpoint(
		checkpoint_path=checkpoint_path,
		tokenizer=tokenizer,
		device=device,
	)

	records = load_coco_image_to_refs(captions_json)
	if limit is not None:
		records = records[: int(limit)]

	all_refs_tok: List[List[List[str]]] = []
	all_hyp_tok: List[List[str]] = []
	samples: List[EvalItem] = []

	for item in tqdm(records, desc="Evaluating", total=len(records)):
		image_path = Path(images_dir) / str(item["file_name"])
		if not image_path.exists():
			continue

		image_tensor = load_image_tensor(image_path=image_path, image_size=image_size).to(dev)
		pred = predict_caption(
			model=model,
			tokenizer=tokenizer,
			image_tensor=image_tensor,
			strategy=strategy,
			beam_size=beam_size,
			top_k=top_k,
			max_len=max_len,
		)

		refs = [str(x) for x in item["references"]]
		refs_tok = [tokenize_caption(x) for x in refs]
		hyp_tok = tokenize_caption(pred.caption)

		if refs_tok and hyp_tok:
			all_refs_tok.append(refs_tok)
			all_hyp_tok.append(hyp_tok)

		samples.append(
			EvalItem(
				image_id=int(item["image_id"]),
				file_name=str(item["file_name"]),
				prediction=str(pred.caption),
				references=refs,
			)
		)

	metrics = compute_bleu_scores(all_refs_tok, all_hyp_tok) if all_hyp_tok else {
		"bleu1": 0.0,
		"bleu2": 0.0,
		"bleu3": 0.0,
		"bleu4": 0.0,
	}
	metrics["num_images"] = int(len(all_hyp_tok))
	metrics["pred_len_avg"] = mean_caption_len(all_hyp_tok)
	return metrics, samples
