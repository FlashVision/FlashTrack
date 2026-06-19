# Quick Start

## Track Objects in a Video

```python
from flashtrack import Predictor

predictor = Predictor(
    model_path="workspace/model_best_inference.pth",
    tracker_type="bytetrack",
)
predictor.track_video("input.mp4", output_dir="output/")
```

## Use a Tracker Directly

```python
import numpy as np
from flashtrack import ByteTracker

tracker = ByteTracker(track_thresh=0.5, track_buffer=30)

# Detections: [x1, y1, x2, y2, score]
detections = np.array([
    [100, 100, 200, 200, 0.95],
    [300, 300, 400, 400, 0.87],
])

tracks = tracker.update(detections)
# tracks: [x1, y1, x2, y2, track_id, class_id, score]
for t in tracks:
    print(f"Track {int(t[4])}: ({t[0]:.0f}, {t[1]:.0f}) - ({t[2]:.0f}, {t[3]:.0f})")
```

## Train a ReID Model

```python
from flashtrack import Trainer

trainer = Trainer(
    model_size="m",
    epochs=120,
    train_data="data/MOT17/train",
    amp=True,
)
trainer.train()
```

## CLI Usage

```bash
# Check installation
flashtrack check

# Train
flashtrack train --train-data data/MOT17/train --model-size m

# Track video
flashtrack track --source video.mp4 --tracker bytetrack

# Export to ONNX
flashtrack export --model workspace/model_best_inference.pth
```
