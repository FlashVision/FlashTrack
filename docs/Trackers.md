# Trackers

FlashTrack provides three tracking algorithms.

## ByteTracker

Two-stage IoU-based tracker from ByteTrack (Zhang et al., 2022).

- **Stage 1**: Match high-confidence detections to tracks via IoU
- **Stage 2**: Match remaining tracks to low-confidence detections

```python
from flashtrack import ByteTracker

tracker = ByteTracker(
    track_thresh=0.5,    # High/low confidence split
    track_buffer=30,     # Frames to keep lost tracks
    match_thresh=0.8,    # IoU threshold
)
tracks = tracker.update(detections)  # [N, 5] -> [M, 7]
```

**Best for**: Speed-critical applications, when ReID features are not available.

## SORTTracker

Classic SORT algorithm (Bewley et al., 2016): Kalman filter + Hungarian + IoU.

```python
from flashtrack import SORTTracker

tracker = SORTTracker(
    max_age=30,         # Max frames without update
    min_hits=3,         # Min hits for confirmation
    iou_threshold=0.3,  # IoU matching threshold
)
tracks = tracker.update(detections)
```

**Best for**: Simple tracking scenarios with minimal occlusion.

## DeepSORTTracker

Deep SORT (Wojke et al., 2017): SORT + ReID cosine distance matching.

```python
from flashtrack import DeepSORTTracker

tracker = DeepSORTTracker(
    max_age=70,
    max_cosine_distance=0.4,
    gallery_size=100,
)
tracks = tracker.update(detections, features=reid_embeddings)
```

**Best for**: Crowded scenes, frequent occlusions, re-identification after disappearance.

## Output Format

All trackers return `[M, 7]` arrays:
```
[x1, y1, x2, y2, track_id, class_id, score]
```

## Comparison

| Tracker | ReID | Occlusion | Speed | Use Case |
|---|---|---|---|---|
| ByteTracker | No | Medium | Fastest | General purpose |
| SORTTracker | No | Low | Fast | Simple scenes |
| DeepSORTTracker | Yes | High | Moderate | Crowded scenes |
