# Training

## Dataset Format

FlashTrack expects MOT17/MOT20-format datasets:

```
data/MOT17/train/
  MOT17-02-DPM/
    img1/
      000001.jpg
      000002.jpg
      ...
    gt/
      gt.txt
  MOT17-04-DPM/
    ...
```

Each line in `gt.txt`:
```
frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
```

## Basic Training

```python
from flashtrack import Trainer

trainer = Trainer(
    model_size="m",
    epochs=120,
    batch_size=64,
    lr=0.0003,
    train_data="data/MOT17/train",
    val_data="data/MOT17/val",
    amp=True,
    save_dir="workspace/reid_training",
)
results = trainer.train()
```

## YAML Config Training

```bash
flashtrack train --config configs/flashtrack_m.yaml
```

## Training Outputs

```
workspace/reid_training/
  checkpoint_best.pth        # Best checkpoint (full state)
  checkpoint_last.pth        # Latest checkpoint
  model_best_inference.pth   # Best model (inference weights)
  model_final_inference.pth  # Final model (FP32)
  model_final_fp16.pth       # Final model (FP16)
  train_*.log                # Training log
```

## Loss Functions

- **Triplet loss** — Hard positive/negative mining for embedding learning
- **Classification loss** — Cross-entropy for identity prediction
- Combined: `total = triplet_weight * triplet + cls_weight * classification`

## Mixed Precision

Enable AMP for faster training with lower memory:

```python
trainer = Trainer(amp=True, ...)
```

## Resume Training

```python
trainer = Trainer(resume="workspace/checkpoint_last.pth", ...)
```
