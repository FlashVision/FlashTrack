"""Tests for BoT-SORT, OC-SORT, OSTrack, HOTA, and CMC."""

import numpy as np
import torch


def test_bot_sort_basic():
    from flashtrack.trackers.bot_sort import BoTSORTTracker

    tracker = BoTSORTTracker(
        track_thresh=0.3,
        track_buffer=30,
        match_thresh=0.8,
        cmc_method=None,
    )

    dets1 = np.array(
        [
            [100, 100, 100, 100],
            [300, 300, 100, 100],
        ],
        dtype=np.float32,
    )
    scores1 = np.array([0.9, 0.8], dtype=np.float32)
    tracks1 = tracker.update(dets1, scores1)
    assert isinstance(tracks1, list)

    dets2 = np.array(
        [
            [105, 105, 100, 100],
            [305, 305, 100, 100],
        ],
        dtype=np.float32,
    )
    scores2 = np.array([0.9, 0.85], dtype=np.float32)
    tracks2 = tracker.update(dets2, scores2)
    assert isinstance(tracks2, list)

    tracker.reset()


def test_bot_sort_with_features():
    from flashtrack.trackers.bot_sort import BoTSORTTracker

    tracker = BoTSORTTracker(
        track_thresh=0.3,
        lambda_iou=0.5,
        lambda_app=0.5,
        cmc_method=None,
    )

    dets = np.array([[100, 100, 100, 100]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    features = np.random.randn(1, 128).astype(np.float32)
    tracks = tracker.update(dets, scores, features=features)
    assert isinstance(tracks, list)

    tracker.reset()


def test_bot_sort_empty():
    from flashtrack.trackers.bot_sort import BoTSORTTracker

    tracker = BoTSORTTracker(cmc_method=None)
    dets = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    tracks = tracker.update(dets, scores)
    assert len(tracks) == 0


def test_oc_sort_basic():
    from flashtrack.trackers.oc_sort import OCSORTTracker

    tracker = OCSORTTracker(max_age=30, min_hits=1, iou_threshold=0.3)

    dets = np.array(
        [
            [50, 50, 100, 100],
            [200, 200, 100, 100],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.95, 0.85], dtype=np.float32)
    tracks = tracker.update(dets, scores)
    assert isinstance(tracks, list)

    dets2 = np.array(
        [
            [55, 55, 100, 100],
            [205, 205, 100, 100],
        ],
        dtype=np.float32,
    )
    scores2 = np.array([0.9, 0.8], dtype=np.float32)
    tracks2 = tracker.update(dets2, scores2)
    assert isinstance(tracks2, list)

    tracker.reset()


def test_oc_sort_ocr():
    from flashtrack.trackers.oc_sort import OCSORTTracker

    tracker = OCSORTTracker(
        max_age=30,
        min_hits=1,
        iou_threshold=0.3,
        use_ocm=True,
        use_ocr=True,
    )

    for frame in range(5):
        dets = np.array([[100 + frame * 5, 100, 100, 100]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        tracker.update(dets, scores)

    assert len(tracker.tracks) > 0
    tracker.reset()


def test_oc_sort_empty():
    from flashtrack.trackers.oc_sort import OCSORTTracker

    tracker = OCSORTTracker()
    dets = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)
    tracks = tracker.update(dets, scores)
    assert len(tracks) == 0


def test_ostrack_forward():
    from flashtrack.trackers.sot.ostrack import OSTrack

    model = OSTrack(
        template_size=64,
        search_size=128,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
        eliminate_layer=-1,
    )
    model.eval()

    template = torch.randn(1, 3, 64, 64)
    search = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(template, search)
    assert "center" in out
    assert "size" in out
    assert "offset" in out
    assert out["center"].shape[0] == 1


def test_ostrack_with_elimination():
    from flashtrack.trackers.sot.ostrack import OSTrack

    model = OSTrack(
        template_size=64,
        search_size=128,
        patch_size=8,
        embed_dim=64,
        depth=4,
        num_heads=4,
        eliminate_layer=2,
        keep_ratio=0.5,
    )
    model.eval()

    template = torch.randn(1, 3, 64, 64)
    search = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(template, search)
    assert "center" in out


def test_ostrack_predict_bbox():
    from flashtrack.trackers.sot.ostrack import OSTrack

    model = OSTrack(
        template_size=64,
        search_size=128,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    template = torch.randn(1, 3, 64, 64)
    search = torch.randn(1, 3, 128, 128)
    bbox = model.predict_bbox(template, search, (50, 50, 200, 200))
    assert len(bbox) == 4
    x, y, w, h = bbox
    assert isinstance(x, float)


def test_hota_perfect():
    from flashtrack.analytics.hota import compute_hota

    gt_boxes = [np.array([[10, 10, 50, 50], [60, 60, 100, 100]])]
    gt_ids = [np.array([1, 2])]
    pred_boxes = [np.array([[10, 10, 50, 50], [60, 60, 100, 100]])]
    pred_ids = [np.array([1, 2])]

    result = compute_hota(gt_boxes, gt_ids, pred_boxes, pred_ids)
    assert "HOTA" in result
    assert "DetA" in result
    assert "AssA" in result
    assert "LocA" in result
    assert result["DetA"] > 0.5
    assert result["HOTA"] > 0.5


def test_hota_empty():
    from flashtrack.analytics.hota import compute_hota

    result = compute_hota(
        [np.empty((0, 4))],
        [np.array([])],
        [np.empty((0, 4))],
        [np.array([])],
    )
    assert result["HOTA"] == 0.0


def test_hota_summary():
    from flashtrack.analytics.hota import compute_hota_summary

    gt_boxes = [np.array([[10, 10, 50, 50]])]
    gt_ids = [np.array([1])]
    pred_boxes = [np.array([[10, 10, 50, 50]])]
    pred_ids = [np.array([1])]

    summary = compute_hota_summary(gt_boxes, gt_ids, pred_boxes, pred_ids)
    assert "HOTA" in summary
    assert "DetA" in summary


def test_cmc_identity():
    from flashtrack.utils.cmc import CameraMotionCompensator

    cmc = CameraMotionCompensator(method="none")
    frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    warp = cmc.compute(frame)
    assert warp.shape == (2, 3)


def test_cmc_numpy_affine():
    from flashtrack.utils.cmc import CameraMotionCompensator

    cmc = CameraMotionCompensator(method="affine")
    frame1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    frame2 = frame1.copy()

    warp1 = cmc.compute(frame1)
    assert warp1.shape == (2, 3)

    warp2 = cmc.compute(frame2)
    assert warp2.shape == (2, 3)

    cmc.reset()


def test_cmc_apply_to_boxes():
    from flashtrack.utils.cmc import CameraMotionCompensator

    cmc = CameraMotionCompensator(method="none")
    boxes = np.array([[10, 20, 50, 60], [100, 200, 30, 40]], dtype=np.float64)
    warp = np.eye(2, 3, dtype=np.float64)
    warp[0, 2] = 5  # translate x by 5
    warp[1, 2] = 10  # translate y by 10

    transformed = cmc.apply_to_boxes(boxes, warp)
    assert transformed.shape == boxes.shape
    np.testing.assert_allclose(transformed[0, 0], 15.0, atol=0.1)
    np.testing.assert_allclose(transformed[0, 1], 30.0, atol=0.1)


def test_compose_and_invert_warp():
    from flashtrack.utils.cmc import compose_warp, invert_warp

    warp1 = np.eye(2, 3, dtype=np.float64)
    warp1[0, 2] = 10
    warp2 = np.eye(2, 3, dtype=np.float64)
    warp2[1, 2] = 20

    composed = compose_warp(warp1, warp2)
    assert composed.shape == (2, 3)
    np.testing.assert_allclose(composed[0, 2], 10.0, atol=0.1)
    np.testing.assert_allclose(composed[1, 2], 20.0, atol=0.1)

    inv = invert_warp(warp1)
    assert inv.shape == (2, 3)
    np.testing.assert_allclose(inv[0, 2], -10.0, atol=0.1)
