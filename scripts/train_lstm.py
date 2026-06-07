"""
- loads COCO captions dataset
- builds ResNet encoder + LSTM decoder
- runs teacher-forcing training
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.coco_dataset import CocoCaptionDataset, CocoPaths
from src.datasets.collate import coco_collate_fn
from src.datasets.tokenizer import CaptionTokenizer
from src.models.captioner import Captioner
from src.models.decoder_lstm import LSTMDecoder
from src.models.encoder_resnet import ResNetEncoder
from src.training.optim import build_optimizer
from src.training.trainer import TrainConfig, Trainer
from src.utils.seed import seed_everything


def _load_resume_checkpoint(
    resume_path: Path,
    model: Captioner,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    optimizer_state = ckpt.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    start_step = int(ckpt.get("step", 0))
    return start_epoch, start_step


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=str, default="data/raw/coco2017/train2017")
    ap.add_argument(
        "--captions-json",
        type=str,
        default="data/raw/coco2017/annotations/captions_train2017.json",
    )
    ap.add_argument("--tokenizer-dir", type=str, default="data/processed/tokenizer")

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="Limit samples for debugging/overfit; 0 uses the full dataset.")
    ap.add_argument(
        "--one-caption-per-image",
        action="store_true",
        help="Sample one caption per image each epoch to avoid repeated encoder passes.",
    )

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-dir", type=str, default="outputs/checkpoints")
    ap.add_argument("--save-every-steps", type=int, default=0)
    ap.add_argument("--resume-from", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--encoder", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--encoder-trainable", action="store_true")
    ap.add_argument("--proj-dim", type=int, default=512)

    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument(
        "--overfit-batches",
        type=int,
        default=0,
        help="If >0, only run this many batches each epoch (useful for quick sanity).",
    )

    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    seed_everything(int(args.seed))

    tok = CaptionTokenizer.load(Path(args.tokenizer_dir), max_len=int(args.max_len))

    paths = CocoPaths(
        images_dir=Path(args.images_dir),
        captions_json=Path(args.captions_json),
        # CocoCaptionDataset expects a HuggingFace tokenizer directory (save_pretrained format)
        vocab_json=Path(args.tokenizer_dir),
    )

    ds = CocoCaptionDataset(
        paths=paths,
        max_len=int(args.max_len),
        limit=int(args.limit) if args.limit else None,
        one_caption_per_image=bool(args.one_caption_per_image),
    )
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=coco_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # HuggingFace tokenizers can have added tokens; len(tokenizer) reflects the full embedding size.
    vocab_size = len(tok.tokenizer)

    encoder = ResNetEncoder(
        name=str(args.encoder),
        pretrained=True,
        trainable=bool(args.encoder_trainable),
        proj_dim=int(args.proj_dim),
    )
    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        encoder_dim=encoder.out_dim,
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        pad_id=int(tok.pad_id),
    )
    model = Captioner(encoder=encoder, decoder=decoder)

    optimizer = build_optimizer(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    cfg = TrainConfig(
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip),
        device=str(args.device),
        log_every=int(args.log_every),
        save_dir=str(args.save_dir),
        save_every_steps=int(args.save_every_steps),
        overfit_batches=int(args.overfit_batches),
    )

    checkpoint_extra = {
        "model_config": {
            "encoder": {
                "name": str(args.encoder),
                "pretrained": True,
                "trainable": bool(args.encoder_trainable),
                "proj_dim": int(args.proj_dim),
            },
            "decoder": {
                "embed_dim": int(args.embed_dim),
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.num_layers),
                "dropout": float(args.dropout),
            },
        },
        "tokenizer_dir": str(Path(args.tokenizer_dir)),
        "max_len": int(args.max_len),
        "one_caption_per_image": bool(args.one_caption_per_image),
    }

    start_epoch = 0
    start_step = 0
    if args.resume_from:
        start_epoch, start_step = _load_resume_checkpoint(
            resume_path=Path(args.resume_from),
            model=model,
            optimizer=optimizer,
        )
        print(f"[OK] resumed from {args.resume_from} at epoch={start_epoch} step={start_step}")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        pad_id=int(tok.pad_id),
        checkpoint_extra=checkpoint_extra,
    )
    trainer.fit(dl, start_epoch=start_epoch, start_step=start_step)


if __name__ == "__main__":
    main()
