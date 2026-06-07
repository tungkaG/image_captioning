from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description="Download a size-capped COCO image subset.")
	ap.add_argument("--split", choices=["train2017", "val2017"], default="train2017")
	ap.add_argument("--annotations-json", type=str, default="")
	ap.add_argument("--out-dir", type=str, default="")
	ap.add_argument("--max-bytes-gb", type=float, default=5.0)
	ap.add_argument("--max-images", type=int, default=0)
	ap.add_argument("--retries", type=int, default=5)
	ap.add_argument("--retry-delay", type=float, default=2.0)
	ap.add_argument("--timeout", type=float, default=60.0)
	ap.add_argument("--report-every", type=int, default=100)
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--base-url", type=str, default="http://images.cocodataset.org")
	return ap.parse_args()


def default_annotations_path(split: str) -> Path:
	if split == "train2017":
		return Path("data/raw/coco2017/annotations/captions_train2017.json")
	return Path("data/raw/coco2017/annotations/captions_val2017.json")


def default_output_dir(split: str, max_bytes_gb: float) -> Path:
	rounded = str(max_bytes_gb).replace(".", "p")
	return Path("data/raw/coco2017") / f"{split}_subset_{rounded}gb"


def load_target_images(annotations_json: Path) -> list[str]:
	payload = json.loads(annotations_json.read_text(encoding="utf-8"))
	caption_image_ids = {ann["image_id"] for ann in payload["annotations"]}
	ordered = [img["file_name"] for img in payload["images"] if img["id"] in caption_image_ids]
	return ordered


def validate_image_file(path: Path) -> int:
	with Image.open(path) as image:
		image.load()
	return int(path.stat().st_size)


def existing_file_sizes(out_dir: Path) -> dict[str, int]:
	if not out_dir.exists():
		return {}

	existing: dict[str, int] = {}
	for path in out_dir.iterdir():
		if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
			continue
		try:
			existing[path.name] = validate_image_file(path)
		except (OSError, ValueError) as exc:
			print(f"[WARN] removing invalid existing file {path.name}: {exc}")
			path.unlink(missing_ok=True)
	return existing


def download_file(url: str, destination: Path, timeout: float, retries: int, retry_delay: float) -> int:
	destination.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = destination.with_suffix(destination.suffix + ".part")

	for attempt in range(int(retries) + 1):
		try:
			with urllib.request.urlopen(url, timeout=float(timeout)) as response, tmp_path.open("wb") as handle:
				shutil.copyfileobj(response, handle)
			validate_image_file(tmp_path)
			tmp_path.replace(destination)
			return int(destination.stat().st_size)
		except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
			if tmp_path.exists():
				tmp_path.unlink()
			if attempt >= int(retries):
				raise RuntimeError(f"failed to download {url}: {exc}") from exc
			time.sleep(float(retry_delay))

	raise RuntimeError(f"failed to download {url}")


def write_summary(
	summary_path: Path,
	split: str,
	annotations_json: Path,
	out_dir: Path,
	byte_budget: int,
	image_count: int,
	bytes_on_disk: int,
	failed_downloads: list[str],
) -> None:
	payload = {
		"split": split,
		"annotations_json": str(annotations_json),
		"out_dir": str(out_dir),
		"byte_budget": int(byte_budget),
		"image_count": int(image_count),
		"bytes_on_disk": int(bytes_on_disk),
		"failed_downloads": failed_downloads,
	}
	summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
	args = parse_args()

	split = str(args.split)
	annotations_json = Path(args.annotations_json) if args.annotations_json else default_annotations_path(split)
	out_dir = Path(args.out_dir) if args.out_dir else default_output_dir(split, float(args.max_bytes_gb))
	byte_budget = int(float(args.max_bytes_gb) * 1024 * 1024 * 1024)

	if not annotations_json.exists():
		raise FileNotFoundError(f"annotations file not found: {annotations_json}")

	image_files = load_target_images(annotations_json)
	existing = existing_file_sizes(out_dir)
	bytes_on_disk = sum(existing.values())
	image_count = len(existing)
	failed_downloads: list[str] = []

	print(f"[INFO] split={split}")
	print(f"[INFO] annotations={annotations_json}")
	print(f"[INFO] out_dir={out_dir}")
	print(f"[INFO] budget_bytes={byte_budget}")
	print(f"[INFO] existing_images={image_count} existing_bytes={bytes_on_disk}")

	pending_files = [file_name for file_name in image_files if file_name not in existing]
	next_index = 0
	active_futures: dict[concurrent.futures.Future[int], str] = {}
	max_workers = max(1, int(args.workers))

	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
		while True:
			while len(active_futures) < max_workers and next_index < len(pending_files):
				if bytes_on_disk >= byte_budget:
					break
				if args.max_images and image_count + len(active_futures) >= int(args.max_images):
					break

				file_name = pending_files[next_index]
				next_index += 1
				url = f"{args.base_url.rstrip('/')}/{split}/{file_name}"
				destination = out_dir / file_name
				future = executor.submit(
					download_file,
					url,
					destination,
					float(args.timeout),
					int(args.retries),
					float(args.retry_delay),
				)
				active_futures[future] = file_name

			if not active_futures:
				break

			done, _ = concurrent.futures.wait(
				active_futures.keys(),
				return_when=concurrent.futures.FIRST_COMPLETED,
			)

			for future in done:
				file_name = active_futures.pop(future)
				try:
					file_size = int(future.result())
				except RuntimeError as exc:
					failed_downloads.append(file_name)
					print(f"[WARN] {exc}")
					continue

				existing[file_name] = file_size
				bytes_on_disk += file_size
				image_count += 1

				if image_count % int(args.report_every) == 0:
					print(f"[INFO] images={image_count} bytes={bytes_on_disk} last={file_name}")

			if bytes_on_disk >= byte_budget:
				print("[OK] byte budget reached; stopping download")
				break
			if args.max_images and image_count >= int(args.max_images):
				print("[OK] image limit reached; stopping download")
				break

		for future in active_futures:
			future.cancel()

	write_summary(
		summary_path=out_dir / "subset_summary.json",
		split=split,
		annotations_json=annotations_json,
		out_dir=out_dir,
		byte_budget=byte_budget,
		image_count=image_count,
		bytes_on_disk=bytes_on_disk,
		failed_downloads=failed_downloads,
	)

	print(f"[OK] finished with images={image_count} bytes={bytes_on_disk}")
	print(f"[OK] summary written to {out_dir / 'subset_summary.json'}")


if __name__ == "__main__":
	main()