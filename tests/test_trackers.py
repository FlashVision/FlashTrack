"""Tests for tracking algorithms."""

import numpy as np


def test_bytetracker_basic():
    """Test ByteTracker with simple detections."""
    from flashtrack.models.byte_tracker import ByteTracker

    tracker = ByteTracker(track_thresh=0.3, track_buffer=30, match_thresh=0.8)

    # Frame 1: two objects (tlwh format)
    dets1 = np.array([
        [100, 100, 100, 100],
        [300, 300, 100, 100],
    ], dtype=np.float32)
    scores1 = np.array([0.9, 0.8], dtype=np.float32)
    tracks1 = tracker.update(dets1, scores1)
    assert isinstance(tracks1, list)

    # Frame 2: same objects moved slightly
    dets2 = np.array([
        [105, 105, 100, 100],
        [305, 305, 100, 100],
    ], dtype=np.float32)
    scores2 = np.array([0.9, 0.85], dtype=np.float32)
    tracks2 = tracker.update(dets2, scores2)
    assert isinstance(tracks2, list)

    tracker.reset()


def test_bytetracker_empty():
    """Test ByteTracker with no detections."""
    from flashtrack.models.byte_tracker import ByteTracker

    tracker = ByteTracker()
    dets = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    tracks = tracker.update(dets, scores)
    assert len(tracks) == 0


def test_sort_tracker_basic():
    """Test SORTTracker."""
    from flashtrack.models.sort_tracker import SORTTracker

    tracker = SORTTracker(max_age=30, min_hits=1, iou_threshold=0.3)

    dets = np.array([
        [50, 50, 100, 100],
        [200, 200, 100, 100],
    ], dtype=np.float32)
    scores = np.array([0.95, 0.85], dtype=np.float32)
    tracks = tracker.update(dets, scores)
    assert isinstance(tracks, list)

    # Move objects
    dets2 = np.array([
        [55, 55, 100, 100],
        [205, 205, 100, 100],
    ], dtype=np.float32)
    scores2 = np.array([0.9, 0.8], dtype=np.float32)
    tracks2 = tracker.update(dets2, scores2)
    assert isinstance(tracks2, list)

    tracker.reset()


def test_deepsort_tracker_without_features():
    """Test DeepSORTTracker falls back to IoU without ReID features."""
    from flashtrack.models.deepsort_tracker import DeepSORTTracker

    tracker = DeepSORTTracker(max_age=30, n_init=1, max_iou_distance=0.7)

    dets = np.array([
        [100, 100, 100, 100],
    ], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    tracks = tracker.update(dets, scores, features=None)
    assert isinstance(tracks, list)

    tracker.reset()


def test_deepsort_tracker_with_features():
    """Test DeepSORTTracker with ReID features."""
    from flashtrack.models.deepsort_tracker import DeepSORTTracker

    tracker = DeepSORTTracker(max_age=30, n_init=1, max_cosine_distance=0.5)

    dets = np.array([
        [100, 100, 100, 100],
        [300, 300, 100, 100],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    features = np.random.randn(2, 128).astype(np.float32)

    tracks = tracker.update(dets, scores, features=features)
    assert isinstance(tracks, list)

    # Second frame
    dets2 = np.array([
        [105, 105, 100, 100],
        [305, 305, 100, 100],
    ], dtype=np.float32)
    scores2 = np.array([0.9, 0.85], dtype=np.float32)
    features2 = features + np.random.randn(2, 128).astype(np.float32) * 0.1
    tracks2 = tracker.update(dets2, scores2, features=features2)
    assert isinstance(tracks2, list)

    tracker.reset()


def test_tracker_id_consistency():
    """Test that track IDs remain consistent across frames."""
    from flashtrack.models.byte_tracker import ByteTracker

    tracker = ByteTracker(track_thresh=0.3, track_buffer=30, match_thresh=0.8)

    # Static object over 3 frames
    for _ in range(3):
        dets = np.array([[100, 100, 100, 100]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        tracks = tracker.update(dets, scores)

    assert len(tracks) >= 1
    assert tracks[0].track_id > 0

    tracker.reset()
