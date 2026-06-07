from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the COCO subset captioning pipeline with retries.")
    ap.add_argument("--split", choices=["train2017", "val2017"], default="train2017")
    ap.add_argument("--subset-dir", type=str, default="data/raw/coco2017/train2017_subset_5p0gb")
    ap.add_argument("--annotations-json", type=str, default="data/raw/coco2017/annotations/captions_train2017.json")
    ap.add_argument("--val-images-dir", type=str, default="data/raw/coco2017/val2017")
    ap.add_argument("--val-captions-json", type=str, default="data/raw/coco2017/annotations/captions_val2017.json")
    ap.add_argument("--tokenizer-dir", type=str, default="data/processed/tokenizer")
    ap.add_argument("--max-bytes-gb", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report-every", type=int, default=250)
    ap.add_argument("--encoder", type=str, default="resnet18")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every-steps", type=int, default=250)
    ap.add_argument("--save-dir", type=str, default="outputs/checkpoints/train_subset5gb_resnet18")
    ap.add_argument("--beam-size", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--eval-limit", type=int, default=200)
    ap.add_argument("--show-samples", type=int, default=5)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-delay", type=float, default=5.0)
    return ap.parse_args()


def run_command(args: list[str]) -> None:
    print(f"[RUN] {shlex.join(args)}", flush=True)
    subprocess.run(args, check=True)


def run_with_retries(name: str, factory, max_retries: int, retry_delay: float) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, int(max_retries) + 1):
        try:
            print(f"[STEP] {name} attempt={attempt}/{max_retries}", flush=True)
            run_command(factory())
            print(f"[OK] {name} completed", flush=True)
            return
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            print(f"[WARN] {name} failed with exit code {exc.returncode}", flush=True)
            if attempt >= int(max_retries):
                break
            time.sleep(float(retry_delay))

    if last_exc is None:
        raise RuntimeError(f"{name} failed without an exception")
    raise last_exc


def main() -> None:
    args = parse_args()
    python_exe = sys.executable
    subset_dir = Path(args.subset_dir)
    save_dir = Path(args.save_dir)
    latest_ckpt = save_dir / "captioner_latest.pt"

    def download_factory() -> list[str]:
        return [
            python_exe,
            "scripts/download_coco.py",
            "--split",
            str(args.split),
            "--out-dir",
            str(subset_dir),
            "--max-bytes-gb",
            str(args.max_bytes_gb),
            "--report-every",
            str(args.report_every),
            "--workers",
            str(args.workers),
        ]

    def train_factory() -> list[str]:
        cmd = [
            python_exe,
            "scripts/train_lstm.py",
            "--images-dir",
            str(subset_dir),
            "--captions-json",
            str(args.annotations_json),
            "--tokenizer-dir",
            str(args.tokenizer_dir),
            "--encoder",
            str(args.encoder),
            "--batch-size",
            str(args.batch_size),
            "--one-caption-per-image",
            "--epochs",
            str(args.epochs),
            "--log-every",
            str(args.log_every),
            "--save-every-steps",
            str(args.save_every_steps),
            "--save-dir",
            str(save_dir),
        ]
        if latest_ckpt.exists():
            cmd.extend(["--resume-from", str(latest_ckpt)])
        return cmd

    def eval_factory() -> list[str]:
        return [
            python_exe,
            "scripts/eval.py",
            "--ckpt",
            str(latest_ckpt),
            "--tokenizer-dir",
            str(args.tokenizer_dir),
            "--images-dir",
            str(args.val_images_dir),
            "--captions-json",
            str(args.val_captions_json),
            "--strategy",
            "beam",
            "--beam-size",
            str(args.beam_size),
            "--limit",
            str(args.eval_limit),
            "--show-samples",
            str(args.show_samples),
        ]

    def predict_factory() -> list[str]:
        return [
            python_exe,
            "scripts/predict_dataset.py",
            "--ckpt",
            str(latest_ckpt),
            "--tokenizer-dir",
            str(args.tokenizer_dir),
            "--images-dir",
            str(args.val_images_dir),
            "--captions-json",
            str(args.val_captions_json),
            "--random",
            "--strategy",
            "beam",
            "--beam-size",
            str(args.beam_size),
            "--top-k",
            str(args.top_k),
        ]

    run_with_retries("download subset", download_factory, max_retries=args.max_retries, retry_delay=args.retry_delay)
    run_with_retries("train model", train_factory, max_retries=args.max_retries, retry_delay=args.retry_delay)
    run_with_retries("evaluate model", eval_factory, max_retries=args.max_retries, retry_delay=args.retry_delay)
    run_with_retries("predict sample", predict_factory, max_retries=args.max_retries, retry_delay=args.retry_delay)


if __name__ == "__main__":
    main()