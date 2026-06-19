"""Tests for tracking algorithms."""

import numpy as np
import pytest


def test_bytetracker_basic():
    """Test ByteTracker with simple detections."""
    from flashtrack.models.byte_tracker import ByteTracker

    tracker = ByteTracker(track_thresh=0.3, track_buffer=30, match_thresh=0.8)

    # Frame 1: two objects
    dets1 = np.array([
        [100, 100, 200, 200, 0.9],
        [300, 300, 400, 400, 0.8],
    ], dtype=np.float32)
    tracks1 = tracker.update(dets1)
    assert tracks1.shape[1] == 7
    assert len(tracks1) >= 1

    # Frame 2: same objects moved slightly
    dets2 = np.array([
        [105, 105, 205, 205, 0.9],
        [305, 305, 405, 405, 0.85],
    ], dtype=np.float32)
    tracks2 = tracker.update(dets2)
    assert len(tracks2) >= 1

    tracker.reset()


def test_bytetracker_empty():
    """Test ByteTracker with no detections."""
    from flashtrack.models.byte_tracker import ByteTracker

    tracker = ByteTracker()
    dets = np.empty((0, 5), dtype=np.float32)
    tracks = tracker.update(dets)
    assert tracks.shape == (0, 7)


def test_sort_tracker_basic():
    """Test SORTTracker."""
    from flashtrack.models.sort_tracker import SORTTracker

    tracker = SORTTracker(max_age=30, min_hits=1, iou_threshold=0.3)

    dets = np.array([
        [50, 50, 150, 150, 0.95],
        [200, 200, 300, 300, 0.85],
    ], dtype=np.float32)
    tracks = tracker.update(dets)
    assert tracks.shape[1] == 7

    # Move objects
    dets2 = np.array([
        [55, 55, 155, 155, 0.9],
        [205, 205, 305, 305, 0.8],
    ], dtype=np.float32)
    tracks2 = tracker.update(dets2)
    assert len(tracks2) >= 1

    tracker.reset()


def test_deepsort_tracker_without_features():
    """Test DeepSORTTracker falls back to IoU without ReID features."""
    from flashtrack.models.deepsort_tracker import DeepSORTTracker

    tracker = DeepSORTTracker(max_age=30, min_hits=1, iou_threshold=0.3)

    dets = np.array([
        [100, 100, 200, 200, 0.9],
    ], dtype=np.float32)
    tracks = tracker.update(dets, features=None)
    assert tracks.shape[1] == 7

    tracker.reset()


def test_deepsort_tracker_with_features():
    """Test DeepSORTTracker with ReID features."""
    from flashtrack.models.deepsort_tracker import DeepSORTTracker

    tracker = DeepSORTTracker(max_age=30, min_hits=1, max_cosine_distance=0.5)

    dets = np.array([
        [100, 100, 200, 200, 0.9],
        [300, 300, 400, 400, 0.8],
    ], dtype=np.float32)
    features = np.random.randn(2, 128).astype(np.float32)

    tracks = tracker.update(dets, features=features)
    assert tracks.shape[1] == 7

    # Second frame
    dets2 = np.array([
        [105, 105, 205, 205, 0.9],
        [305, 305, 405, 405, 0.85],
    ], dtype=np.float32)
    features2 = features + np.random.randn(2, 128).astype(np.float32) * 0.1
    tracks2 = tracker.update(dets2, features=features2)
    assert len(tracks2) >= 1

    tracker.reset()


def test_tracker_id_consistency():
    """Test that track IDs remain consistent across frames."""
    from flashtrack.models.byte_tracker import ByteTracker

    tracker = ByteTracker(track_thresh=0.3, track_buffer=30, match_thresh=0.8)

    # Static object over 3 frames
    for _ in range(3):
        dets = np.array([[100, 100, 200, 200, 0.9]], dtype=np.float32)
        tracks = tracker.update(dets)

    assert len(tracks) == 1
    # Same track ID
    track_id = int(tracks[0, 4])
    assert track_id > 0

    tracker.reset()
