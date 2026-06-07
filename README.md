# Image Captioning on COCO (ResNet + LSTM)

End-to-end image captioning pipeline using:
- ResNet encoder (ImageNet pretrained)
- LSTM decoder (teacher forcing during training)
- Greedy and beam search decoding
- BLEU-based evaluation on COCO validation set

This repository is complete up to training + evaluation + dataset-level prediction vs ground truth.

## 1) Setup

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## 2) Data layout expected

Put COCO data in:

```text
data/raw/coco2017/
   train2017/
   val2017/
   annotations/
      captions_train2017.json
      captions_val2017.json
```

You can also point `--images-dir` at `train2017.zip` or `val2017.zip` directly if you do not want to extract the archives.

You can use [scripts/download_coco.py](scripts/download_coco.py) to fetch a Kaggle mirror.

## 3) Preprocess tokenizer

```bash
python scripts/preprocess_captions.py --model-name gpt2 --out-tokenizer data/processed/tokenizer --max-len 30
```

## 4) Sanity-check dataset pipeline

```bash
python scripts/debug_dataset.py --images-dir data/raw/coco2017/train2017 --captions-json data/raw/coco2017/annotations/captions_train2017.json --tokenizer-dir data/processed/tokenizer --limit 64
```

## 5) Train ResNet+LSTM

Quick debug/overfit style run:

```bash
python scripts/train_lstm.py --images-dir data/raw/coco2017/train2017 --captions-json data/raw/coco2017/annotations/captions_train2017.json --tokenizer-dir data/processed/tokenizer --batch-size 16 --limit 2000 --epochs 2 --log-every 20 --save-dir outputs/checkpoints
```

For CPU-bound training, add `--one-caption-per-image` to sample one caption per image each epoch and avoid repeating the same ResNet forward pass for every COCO caption.

For long runs, add `--save-every-steps 500` to maintain `captioner_latest.pt` during the epoch, and use `--resume-from outputs/checkpoints/.../captioner_latest.pt` to continue from the last saved step.

Checkpoints are saved as:
- outputs/checkpoints/captioner_epoch0.pt
- outputs/checkpoints/captioner_epoch1.pt

## 6) Evaluate on validation set

This computes BLEU-1/2/3/4 and prints qualitative samples with prediction + ground truth captions.

```bash
python scripts/eval.py --ckpt outputs/checkpoints/captioner_epoch1.pt --tokenizer-dir data/processed/tokenizer --images-dir data/raw/coco2017/val2017 --captions-json data/raw/coco2017/annotations/captions_val2017.json --strategy beam --beam-size 3 --limit 200 --show-samples 5
```

## 7) Predict one dataset image and compare to GT

Pick one image from COCO val and print caption + references.

```bash
python scripts/predict_dataset.py --ckpt outputs/checkpoints/captioner_epoch1.pt --tokenizer-dir data/processed/tokenizer --images-dir data/raw/coco2017/val2017 --captions-json data/raw/coco2017/annotations/captions_val2017.json --random --strategy beam --beam-size 5 --top-k 3
```

You can also choose a fixed image by id:

```bash
python scripts/predict_dataset.py --ckpt outputs/checkpoints/captioner_epoch1.pt --image-id 391895
```

## 8) Implemented modules

- Training loop + checkpointing: [src/training/trainer.py](src/training/trainer.py)
- ResNet encoder: [src/models/encoder_resnet.py](src/models/encoder_resnet.py)
- LSTM decoder: [src/models/decoder_lstm.py](src/models/decoder_lstm.py)
- Greedy decoding: [src/inference/greedy.py](src/inference/greedy.py)
- Beam decoding: [src/inference/beam_search.py](src/inference/beam_search.py)
- Prediction utilities: [src/inference/predict.py](src/inference/predict.py)
- BLEU metrics: [src/eval/metrics.py](src/eval/metrics.py)
- Evaluation pipeline: [src/eval/evaluate.py](src/eval/evaluate.py)

## 9) Notes

- CIDEr/METEOR are not yet included in this baseline; BLEU is implemented and ready to run.
- For faster first experiments, use `--limit` in training and eval.
- The encoder is frozen by default; add `--encoder-trainable` in training to fine-tune.