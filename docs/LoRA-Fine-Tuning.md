# LoRA Fine-Tuning

FlashTrack supports 6 LoRA variants for parameter-efficient fine-tuning.

## Available Variants

| Variant | Description | Best For |
|---|---|---|
| `standard` | Classic LoRA (Hu et al., 2022) | General fine-tuning |
| `dora` | Weight-decomposed LoRA (Liu et al., 2024) | Higher quality |
| `lora_plus` | Asymmetric LR for A/B matrices | Faster convergence |
| `adalora` | Adaptive rank via SVD pruning | Automatic rank |
| `ortho` | Orthogonal regularization | Stable training |
| `lora_fa` | Frozen A, trainable B only | Minimal memory |

## Usage

### Python API

```python
from flashtrack import Trainer

trainer = Trainer(
    model_size="m",
    lora=True,
    lora_rank=8,
    lora_variant="standard",
    train_data="data/MOT17/train",
)
trainer.train()
```

### Direct Application

```python
from flashtrack.models.tracker import FlashTracker
from flashtrack.models.lora import apply_lora

model = FlashTracker(backbone_size="1.0x", reid_dim=128, encoder_channels=256)
model = apply_lora(model, rank=8, variant="dora", target_modules=["backbone", "encoder"])
```

### QLoRA (Quantized LoRA)

```python
trainer = Trainer(
    qlora=True,
    qlora_dtype="int8",
    lora_rank=8,
    ...
)
```

## YAML Config

```yaml
train:
  use_lora: true
  lora_rank: 8
  lora_alpha: 16.0
  lora_dropout: 0.05
  lora_target_modules: ["backbone", "encoder"]
```

## Merging LoRA Weights

After training, merge LoRA into base weights for zero-overhead inference:

```python
from flashtrack.models.lora import merge_lora_weights
model = merge_lora_weights(model)
```
