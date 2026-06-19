# FlashTrack

[![CI](https://github.com/FlashVision/FlashTrack/actions/workflows/ci.yml/badge.svg)](https://github.com/FlashVision/FlashTrack/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Ultra-lightweight real-time multi-object tracking** built on ShuffleNetV2 backbone with DeepSORT/ByteTrack-style tracking. Part of the [FlashVision](https://github.com/FlashVision) family (alongside FlashDet, FlashSeg, FlashOCR).

## What is FlashTrack?

FlashTrack combines a lightweight ReID (re-identification) feature extractor with Kalman filtering for robust multi-object tracking. It's designed for edge deployment — small model size, fast inference, and strong tracking accuracy.

**Key features:**
- **Tiny models** — 0.3M to 1.5M parameters (< 3 MB FP16)
- **Three tracker backends** — ByteTracker, SORTTracker, DeepSORTTracker
- **ReID-powered** — Deep appearance features for robust re-identification
- **LoRA fine-tuning** — 6 variants for parameter-efficient adaptation
- **Knowledge Distillation** — Train small models from large teachers
- **ONNX export** — Deploy anywhere

## Model Zoo

| Model | Backbone | ReID Dim | Params | FP16 Size |
|---|---|---|---|---|
| FlashTrack-m-0.5x | ShuffleNetV2 0.5x | 128 | ~0.3M | ~0.6 MB |
| FlashTrack-m | ShuffleNetV2 1.0x | 128 | ~0.8M | ~1.6 MB |
| FlashTrack-m-1.5x | ShuffleNetV2 1.5x | 256 | ~1.5M | ~3.0 MB |

## Trackers

| Tracker | Description | ReID? | Speed |
|---|---|---|---|
| ByteTracker | Two-stage IoU association (ByteTrack) | No | Fastest |
| SORTTracker | Kalman + Hungarian + IoU | No | Fast |
| DeepSORTTracker | Kalman + ReID cosine distance | Yes | Accurate |

## Installation

```bash
# From source
git clone https://github.com/FlashVision/FlashTrack.git
cd FlashTrack
pip install -e ".[all]"

# Verify installation
flashtrack check
```

## Quick Start

### Python API

```python
from flashtrack import FlashTracker, ByteTracker, Predictor

# Run tracking on video
predictor = Predictor(
    model_path="workspace/model_best_inference.pth",
    tracker_type="bytetrack",
)
predictor.track_video("input.mp4", output_dir="output/")

# Use ByteTracker directly
tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
tracks = tracker.update(detections)  # [N, 5] -> [M, 7]
```

### CLI

```bash
# Train ReID model
flashtrack train --train-data data/MOT17/train --model-size m --epochs 120

# Run tracking
flashtrack track --source video.mp4 --tracker bytetrack

# Export to ONNX
flashtrack export --model workspace/model_best_inference.pth --simplify
```

## Training

### Standard Training

```python
from flashtrack import Trainer

trainer = Trainer(
    model_size="m",
    epochs=120,
    batch_size=64,
    train_data="data/MOT17/train",
    val_data="data/MOT17/val",
    amp=True,
)
trainer.train()
```

### LoRA Fine-Tuning

```python
trainer = Trainer(
    model_size="m",
    lora=True,
    lora_rank=8,
    lora_variant="standard",  # or "dora", "lora_plus", "adalora", "ortho", "lora_fa"
    train_data="data/MOT17/train",
)
trainer.train()
```

### Knowledge Distillation

Train a smaller student from a larger teacher:

```bash
flashtrack train --config configs/flashtrack_m_kd.yaml
```

## Architecture

```
Input Crop [B, 3, 128, 64]
    │
    ▼
ShuffleNetV2 Backbone
    │
    ▼
Feature Encoder (1x1 + DW 3x3 + 1x1 → GAP)
    │
    ▼
ReID Head (BN → FC → BN Neck)
    │
    ├── Embedding [B, 128] (inference)
    └── Logits [B, num_ids] (training)
```

## YAML Configuration

```yaml
model:
  backbone_size: "1.0x"
  reid_dim: 128
  num_ids: 750
  input_size: [128, 64]

data:
  train_images: data/MOT17/train
  val_images: data/MOT17/val

train:
  epochs: 120
  batch_size: 64
  learning_rate: 0.0003
```

## License

[MIT License](LICENSE) — free for commercial and academic use.
