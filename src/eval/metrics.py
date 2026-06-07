from __future__ import annotations

from typing import Iterable, List, Sequence

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


def normalize_caption_text(text: str) -> str:
	return " ".join((text or "").strip().lower().split())


def tokenize_caption(text: str) -> List[str]:
	return normalize_caption_text(text).split()


def compute_bleu_scores(
	references: Sequence[Sequence[Sequence[str]]],
	hypotheses: Sequence[Sequence[str]],
) -> dict:
	smooth = SmoothingFunction().method1
	bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
	bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
	bleu3 = corpus_bleu(references, hypotheses, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smooth)
	bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
	return {
		"bleu1": float(bleu1),
		"bleu2": float(bleu2),
		"bleu3": float(bleu3),
		"bleu4": float(bleu4),
	}


def mean_caption_len(items: Iterable[Sequence[str]]) -> float:
	vals = [len(x) for x in items]
	if not vals:
		return 0.0
	return float(sum(vals) / len(vals))
