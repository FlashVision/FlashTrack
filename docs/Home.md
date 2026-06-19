# FlashTrack Documentation

Welcome to the FlashTrack documentation — an ultra-lightweight real-time multi-object tracking framework.

## Overview

FlashTrack is part of the FlashVision family. It combines a lightweight ReID (re-identification) feature extractor built on ShuffleNetV2 with Kalman filtering for robust multi-object tracking.

## Documentation

- [Installation](Installation.md)
- [Quick Start](Quick-Start.md)
- [Models](Models.md)
- [Training](Training.md)
- [Trackers](Trackers.md)
- [LoRA Fine-Tuning](LoRA-Fine-Tuning.md)
- [FAQ](FAQ.md)

## Architecture

```
Input Crop → ShuffleNetV2 → Feature Encoder → ReID Head → Embedding
                                                        └→ Logits (training)
```

## Model Sizes

| Model | Params | FP16 Size | Use Case |
|---|---|---|---|
| m-0.5x | ~0.3M | ~0.6 MB | Ultra-edge (MCU, mobile) |
| m | ~0.8M | ~1.6 MB | General edge |
| m-1.5x | ~1.5M | ~3.0 MB | High accuracy |
