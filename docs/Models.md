# Models

## Architecture

FlashTrack uses a three-stage architecture for ReID feature extraction:

1. **Backbone** — ShuffleNetV2 (0.5x, 1.0x, or 1.5x)
2. **Feature Encoder** — Lightweight CNN that compresses backbone features
3. **ReID Head** — BN neck + FC for embedding, optional classifier for training

## Model Variants

| Model | Backbone | ReID Dim | Encoder Ch | Params | FP16 Size |
|---|---|---|---|---|---|
| FlashTrack-m-0.5x | ShuffleNetV2 0.5x | 128 | 128 | ~0.3M | ~0.6 MB |
| FlashTrack-m | ShuffleNetV2 1.0x | 128 | 256 | ~0.8M | ~1.6 MB |
| FlashTrack-m-1.5x | ShuffleNetV2 1.5x | 256 | 384 | ~1.5M | ~3.0 MB |

## Building a Model

```python
from flashtrack.cfg import get_config
from flashtrack.models import build_model

cfg = get_config(model_size="m", input_size=(128, 64), num_ids=500)
model = build_model(cfg)

# Or directly
from flashtrack.models.tracker import FlashTracker
model = FlashTracker(
    backbone_size="1.0x",
    reid_dim=128,
    encoder_channels=256,
    num_ids=500,
)
```

## Feature Extraction

```python
import torch

model.eval()
crops = torch.randn(4, 3, 128, 64)
embeddings = model.extract_features(crops)  # [4, 128]
```

## Model Info

```python
info = model.get_model_info()
print(f"Parameters: {info['total_params']:,}")
print(f"FP16 size: {info['fp16_mb']:.2f} MB")
```
