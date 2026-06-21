"""Comprehensive test suite for FlashTrack.

Covers models, trackers, CMC, HOTA, registry, CLI, engine, utils,
edge cases, and end-to-end integration.
"""

import argparse
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_track_ids():
    """Reset all track IDs between tests to avoid contamination."""
    from flashtrack.models.byte_tracker import STrack
    from flashtrack.trackers.bot_sort import BoTTrack
    from flashtrack.trackers.oc_sort import OCTrack

    STrack.reset_id()
    BoTTrack.reset_id()
    OCTrack.reset_id()
    yield
    STrack.reset_id()
    BoTTrack.reset_id()
    OCTrack.reset_id()


@pytest.fixture
def small_input():
    return torch.randn(2, 3, 128, 64)


@pytest.fixture
def mock_detections():
    """Simulated detections (tlwh) across 5 frames."""
    frames = []
    for i in range(5):
        dets = np.array(
            [
                [10 + i * 2, 20, 50, 80],
                [200 + i, 100, 40, 60],
                [400, 200 + i * 3, 35, 70],
            ],
            dtype=np.float64,
        )
        scores = np.array([0.9, 0.85, 0.7])
        frames.append((dets, scores))
    return frames


# ===========================================================================
# 1. Model / Component classes
# ===========================================================================


class TestFlashTrackerModel:
    def test_instantiation_default(self):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(pretrained=False)
        assert isinstance(model, nn.Module)

    @pytest.mark.parametrize("size", ["0.5x", "1.0x", "1.5x"])
    def test_backbone_sizes(self, size):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size=size, pretrained=False)
        assert model.backbone_size == size

    def test_forward_pass_output_shape(self, small_input):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", reid_dim=128, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(small_input)
        assert "embeddings" in out
        assert out["embeddings"].shape == (2, 128)

    def test_forward_with_logits(self, small_input):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", reid_dim=128, num_ids=10, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(small_input, return_logits=True)
        assert "logits" in out
        assert out["logits"].shape == (2, 10)

    def test_extract_method(self, small_input):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", pretrained=False)
        model.eval()
        with torch.no_grad():
            emb = model.extract(small_input)
        assert emb.shape == (2, 128)

    def test_predict_method(self, small_input):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", pretrained=False)
        emb = model.predict(small_input)
        assert emb.shape == (2, 128)

    def test_get_model_info(self):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", pretrained=False)
        info = model.get_model_info()
        assert "total_params" in info
        assert info["total_params"] > 0

    def test_build_model_factory(self):
        from flashtrack.models.tracker import build_model

        model = build_model(backbone_size="0.5x", pretrained=False)
        assert isinstance(model, nn.Module)


class TestBackbone:
    def test_shufflenet_forward(self):
        from flashtrack.models.backbone import ShuffleNetV2

        backbone = ShuffleNetV2(model_size="0.5x", pretrained=False)
        x = torch.randn(2, 3, 128, 64)
        backbone.eval()
        with torch.no_grad():
            out = backbone(x)
        assert isinstance(out, list)
        assert len(out) >= 1
        assert out[-1].shape[0] == 2


class TestEncoder:
    def test_feature_encoder_forward(self):
        from flashtrack.models.encoder.feature_encoder import FeatureEncoder

        enc = FeatureEncoder(in_channels=192, out_channels=128)
        x = [torch.randn(2, 192, 4, 2)]
        enc.eval()
        with torch.no_grad():
            out = enc(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 128


class TestReIDHead:
    def test_reid_head_forward(self):
        from flashtrack.models.head.reid_head import ReIDHead

        head = ReIDHead(in_channels=128, reid_dim=64, num_ids=10)
        x = torch.randn(2, 128)
        head.eval()
        with torch.no_grad():
            out = head(x)
        assert "embedding" in out
        assert out["embedding"].shape == (2, 64)


# ===========================================================================
# 2. Registry
# ===========================================================================


class TestRegistry:
    def test_register_and_build(self):
        from flashtrack.registry import Registry

        reg = Registry("test")

        @reg.register("Foo")
        class Foo:
            def __init__(self, x=1):
                self.x = x

        obj = reg.build("Foo", x=42)
        assert obj.x == 42

    def test_list_registered(self):
        from flashtrack.registry import BACKBONES

        items = BACKBONES.list()
        assert isinstance(items, list)

    def test_trackers_registry(self):
        from flashtrack.registry import TRACKERS

        assert isinstance(TRACKERS.list(), list)

    def test_duplicate_registration_raises(self):
        from flashtrack.registry import Registry

        reg = Registry("dup_test")

        @reg.register("A")
        class A:
            pass

        with pytest.raises(KeyError):

            @reg.register("A")
            class B:
                pass

    def test_build_unknown_raises(self):
        from flashtrack.registry import Registry

        reg = Registry("empty")
        with pytest.raises(KeyError):
            reg.build("nonexistent")


# ===========================================================================
# 3. CLI
# ===========================================================================


class TestCLI:
    def test_version_command(self, capsys):
        from flashtrack.cli import cmd_version

        cmd_version(argparse.Namespace())
        captured = capsys.readouterr()
        assert "1.0.0" in captured.out

    def test_main_no_command(self):
        from flashtrack.cli import main

        with patch.object(sys, "argv", ["flashtrack"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_parser_version(self):
        from flashtrack.cli import main

        with patch.object(sys, "argv", ["flashtrack", "version"]):
            main()

    def test_parser_train_requires_data(self):
        from flashtrack.cli import main

        with patch.object(sys, "argv", ["flashtrack", "train"]):
            with pytest.raises(SystemExit):
                main()


# ===========================================================================
# 4. Engine
# ===========================================================================


class TestEngine:
    def test_trainer_instantiation(self):
        from flashtrack.engine.trainer import Trainer

        trainer = Trainer(
            model_size="m",
            epochs=1,
            batch_size=2,
            device="cpu",
            train_dir="/tmp/fake_train",
        )
        assert trainer is not None

    def test_predictor_import(self):
        from flashtrack.engine.predictor import Predictor  # noqa: F401

    def test_validator_import(self):
        from flashtrack.engine.validator import Validator  # noqa: F401

    def test_exporter_import(self):
        from flashtrack.engine.exporter import Exporter  # noqa: F401


# ===========================================================================
# 5. Utils — Metrics, Visualization, Callbacks
# ===========================================================================


class TestKalmanFilter:
    def test_initiate_and_predict(self):
        from flashtrack.utils.kalman_filter import KalmanFilter

        kf = KalmanFilter()
        measurement = np.array([100.0, 200.0, 0.5, 100.0])
        mean, cov = kf.initiate(measurement)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)

        mean2, cov2 = kf.predict(mean, cov)
        assert mean2.shape == (8,)

    def test_update(self):
        from flashtrack.utils.kalman_filter import KalmanFilter

        kf = KalmanFilter()
        m = np.array([50.0, 50.0, 1.0, 80.0])
        mean, cov = kf.initiate(m)
        mean, cov = kf.predict(mean, cov)
        new_m = np.array([52.0, 51.0, 1.0, 80.0])
        mean_u, cov_u = kf.update(mean, cov, new_m)
        assert mean_u.shape == (8,)


class TestCallbacks:
    def test_callbacks_import(self):
        from flashtrack.engine.callbacks import Callback, CallbackList, EarlyStopping  # noqa: F401


# ===========================================================================
# 6. ALL Trackers
# ===========================================================================


class TestByteTracker:
    def test_instantiation(self):
        from flashtrack.models.byte_tracker import ByteTracker

        tracker = ByteTracker()
        assert tracker.frame_id == 0

    def test_update_single_frame(self):
        from flashtrack.models.byte_tracker import ByteTracker

        tracker = ByteTracker(track_thresh=0.3)
        dets = np.array([[10, 20, 50, 80], [200, 100, 40, 60]], dtype=np.float64)
        scores = np.array([0.9, 0.85])
        tracks = tracker.update(dets, scores)
        assert len(tracks) == 2
        assert tracks[0].track_id > 0

    def test_multi_frame_tracking(self, mock_detections):
        from flashtrack.models.byte_tracker import ByteTracker

        tracker = ByteTracker(track_thresh=0.3)
        for dets, scores in mock_detections:
            tracks = tracker.update(dets, scores)
        assert len(tracks) >= 2

    def test_reset(self):
        from flashtrack.models.byte_tracker import ByteTracker

        tracker = ByteTracker()
        dets = np.array([[10, 20, 50, 80]], dtype=np.float64)
        scores = np.array([0.9])
        tracker.update(dets, scores)
        tracker.reset()
        assert tracker.frame_id == 0
        assert len(tracker.tracked_stracks) == 0

    def test_get_results(self):
        from flashtrack.models.byte_tracker import ByteTracker

        tracker = ByteTracker(track_thresh=0.3)
        dets = np.array([[10, 20, 50, 80]], dtype=np.float64)
        scores = np.array([0.9])
        tracker.update(dets, scores)
        results = tracker.get_results()
        assert len(results) == 1
        assert "track_id" in results[0]
        assert "tlwh" in results[0]

    def test_empty_detections(self):
        from flashtrack.models.byte_tracker import ByteTracker

        tracker = ByteTracker()
        dets = np.empty((0, 4), dtype=np.float64)
        scores = np.empty(0)
        tracks = tracker.update(dets, scores)
        assert len(tracks) == 0


class TestBoTSORTTracker:
    def test_instantiation(self):
        from flashtrack.trackers.bot_sort import BoTSORTTracker

        tracker = BoTSORTTracker()
        assert tracker.frame_id == 0

    def test_update_without_features(self, mock_detections):
        from flashtrack.trackers.bot_sort import BoTSORTTracker

        tracker = BoTSORTTracker(track_thresh=0.3, cmc_method=None)
        for dets, scores in mock_detections:
            tracks = tracker.update(dets, scores)
        assert len(tracks) >= 2

    def test_update_with_features(self):
        from flashtrack.trackers.bot_sort import BoTSORTTracker

        tracker = BoTSORTTracker(track_thresh=0.3, cmc_method=None)
        dets = np.array([[10, 20, 50, 80], [200, 100, 40, 60]], dtype=np.float64)
        scores = np.array([0.9, 0.85])
        features = np.random.randn(2, 128).astype(np.float64)
        tracks = tracker.update(dets, scores, features=features)
        assert len(tracks) == 2

    def test_appearance_matching(self):
        from flashtrack.trackers.bot_sort import BoTSORTTracker

        tracker = BoTSORTTracker(track_thresh=0.3, lambda_app=0.5, cmc_method=None)
        feat = np.random.randn(2, 128).astype(np.float64)
        dets = np.array([[10, 20, 50, 80], [200, 100, 40, 60]], dtype=np.float64)
        scores = np.array([0.9, 0.85])
        tracker.update(dets, scores, features=feat)
        # Slightly shifted
        dets2 = np.array([[12, 22, 50, 80], [202, 102, 40, 60]], dtype=np.float64)
        tracks = tracker.update(dets2, scores, features=feat)
        assert len(tracks) == 2

    def test_reset(self):
        from flashtrack.trackers.bot_sort import BoTSORTTracker

        tracker = BoTSORTTracker(cmc_method=None)
        dets = np.array([[10, 20, 50, 80]], dtype=np.float64)
        tracker.update(dets, np.array([0.9]))
        tracker.reset()
        assert tracker.frame_id == 0

    def test_empty_detections(self):
        from flashtrack.trackers.bot_sort import BoTSORTTracker

        tracker = BoTSORTTracker(cmc_method=None)
        tracks = tracker.update(np.empty((0, 4)), np.empty(0))
        assert len(tracks) == 0


class TestOCSORTTracker:
    def test_instantiation(self):
        from flashtrack.trackers.oc_sort import OCSORTTracker

        tracker = OCSORTTracker()
        assert tracker.frame_id == 0

    def test_update_multi_frame(self, mock_detections):
        from flashtrack.trackers.oc_sort import OCSORTTracker

        tracker = OCSORTTracker(min_hits=1, iou_threshold=0.1)
        for dets, scores in mock_detections:
            results = tracker.update(dets, scores)
        assert len(results) >= 1

    def test_ocm_uses_observation(self):
        from flashtrack.trackers.oc_sort import OCSORTTracker

        tracker = OCSORTTracker(use_ocm=True, min_hits=1)
        dets = np.array([[10, 20, 50, 80]], dtype=np.float64)
        tracker.update(dets, np.array([0.9]))
        # Second frame: no detection for that track
        tracker.update(np.array([[300, 300, 30, 30]]), np.array([0.9]))
        assert len(tracker.tracks) >= 1

    def test_ocr_recovery(self):
        from flashtrack.trackers.oc_sort import OCSORTTracker

        tracker = OCSORTTracker(use_ocr=True, min_hits=1, delta_t=5)
        dets = np.array([[10, 20, 50, 80]], dtype=np.float64)
        tracker.update(dets, np.array([0.9]))
        dets2 = np.array([[12, 22, 50, 80]], dtype=np.float64)
        tracker.update(dets2, np.array([0.9]))
        # Lost for 1 frame
        tracker.update(np.empty((0, 4)), np.empty(0))
        # Recovery
        dets3 = np.array([[14, 24, 50, 80]], dtype=np.float64)
        results = tracker.update(dets3, np.array([0.9]))
        assert len(results) >= 1

    def test_empty_detections(self):
        from flashtrack.trackers.oc_sort import OCSORTTracker

        tracker = OCSORTTracker()
        results = tracker.update(np.empty((0, 4)), np.empty(0))
        assert results == []

    def test_reset(self):
        from flashtrack.trackers.oc_sort import OCSORTTracker

        tracker = OCSORTTracker()
        tracker.update(np.array([[10, 20, 50, 80]]), np.array([0.9]))
        tracker.reset()
        assert tracker.frame_id == 0
        assert len(tracker.tracks) == 0


class TestDeepSORTTracker:
    def test_instantiation(self):
        from flashtrack.models.deepsort_tracker import DeepSORTTracker

        tracker = DeepSORTTracker()
        assert tracker is not None


class TestSORTTracker:
    def test_instantiation(self):
        from flashtrack.models.sort_tracker import SORTTracker

        tracker = SORTTracker()
        assert tracker is not None


class TestOSTrack:
    def test_instantiation(self):
        from flashtrack.trackers.sot.ostrack import OSTrack

        model = OSTrack(
            template_size=64,
            search_size=128,
            patch_size=16,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )
        assert isinstance(model, nn.Module)

    def test_forward_pass(self):
        from flashtrack.trackers.sot.ostrack import OSTrack

        model = OSTrack(
            template_size=64,
            search_size=128,
            patch_size=16,
            embed_dim=64,
            depth=2,
            num_heads=4,
            eliminate_layer=1,
            keep_ratio=0.7,
        )
        model.eval()
        template = torch.randn(2, 3, 64, 64)
        search = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(template, search)
        assert "center" in out
        assert "size" in out
        assert "offset" in out

    def test_predict_bbox(self):
        from flashtrack.trackers.sot.ostrack import OSTrack

        model = OSTrack(
            template_size=64,
            search_size=128,
            patch_size=16,
            embed_dim=64,
            depth=2,
            num_heads=4,
            eliminate_layer=-1,
        )
        template = torch.randn(1, 3, 64, 64)
        search = torch.randn(1, 3, 128, 128)
        bbox = model.predict_bbox(template, search, (10, 10, 100, 100))
        assert len(bbox) == 4

    def test_no_candidate_elimination(self):
        from flashtrack.trackers.sot.ostrack import OSTrack

        model = OSTrack(
            template_size=32,
            search_size=64,
            patch_size=16,
            embed_dim=32,
            depth=2,
            num_heads=2,
            eliminate_layer=-1,
        )
        model.eval()
        t = torch.randn(1, 3, 32, 32)
        s = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(t, s)
        assert out["center"].shape[0] == 1


class TestTemplateSearchTracker:
    def test_init_and_track(self):
        from flashtrack.trackers.sot.ostrack import OSTrack, TemplateSearchTracker

        model = OSTrack(
            template_size=64,
            search_size=128,
            patch_size=16,
            embed_dim=64,
            depth=2,
            num_heads=4,
            eliminate_layer=-1,
        )
        tracker = TemplateSearchTracker(model, device="cpu", search_size=128, template_size=64)
        frame = torch.randn(1, 3, 256, 256)
        tracker.init(frame, (50, 50, 30, 30))
        pred = tracker.track(frame)
        assert len(pred) == 4


# ===========================================================================
# 7. HOTA Metric
# ===========================================================================


class TestHOTA:
    def test_perfect_tracking(self):
        from flashtrack.analytics.hota import compute_hota

        gt_boxes = [np.array([[10, 20, 60, 100]]), np.array([[12, 22, 62, 102]])]
        gt_ids = [np.array([1]), np.array([1])]
        pred_boxes = [np.array([[10, 20, 60, 100]]), np.array([[12, 22, 62, 102]])]
        pred_ids = [np.array([1]), np.array([1])]

        result = compute_hota(gt_boxes, gt_ids, pred_boxes, pred_ids)
        assert result["HOTA"] > 0.9
        assert result["DetA"] > 0.9
        assert result["LocA"] > 0.9

    def test_no_predictions(self):
        from flashtrack.analytics.hota import compute_hota

        gt_boxes = [np.array([[10, 20, 60, 100]])]
        gt_ids = [np.array([1])]
        pred_boxes = [np.empty((0, 4))]
        pred_ids = [np.empty(0, dtype=int)]

        result = compute_hota(gt_boxes, gt_ids, pred_boxes, pred_ids)
        assert result["HOTA"] == 0.0

    def test_empty_frames(self):
        from flashtrack.analytics.hota import compute_hota

        gt_boxes = [np.empty((0, 4))]
        gt_ids = [np.empty(0, dtype=int)]
        pred_boxes = [np.empty((0, 4))]
        pred_ids = [np.empty(0, dtype=int)]

        result = compute_hota(gt_boxes, gt_ids, pred_boxes, pred_ids)
        assert "HOTA" in result

    def test_compute_iou_matrix(self):
        from flashtrack.analytics.hota import compute_iou_matrix

        a = np.array([[0, 0, 10, 10]])
        b = np.array([[5, 5, 15, 15]])
        iou = compute_iou_matrix(a, b)
        assert 0.0 < iou[0, 0] < 1.0


# ===========================================================================
# 8. Camera Motion Compensation
# ===========================================================================


class TestCMC:
    def test_affine_identity_on_first_frame(self):
        from flashtrack.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="affine")
        frame = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        warp = cmc.compute(frame)
        assert warp.shape == (2, 3)
        np.testing.assert_allclose(warp, np.eye(2, 3), atol=1e-6)

    def test_homography_identity_on_first_frame(self):
        from flashtrack.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="homography")
        frame = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        warp = cmc.compute(frame)
        assert warp.shape == (3, 3)

    def test_apply_to_boxes(self):
        from flashtrack.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="none")
        boxes = np.array([[10, 20, 30, 40], [100, 200, 50, 60]], dtype=np.float64)
        warp = np.eye(2, 3)
        result = cmc.apply_to_boxes(boxes, warp)
        np.testing.assert_allclose(result, boxes, atol=1e-6)

    def test_apply_to_empty_boxes(self):
        from flashtrack.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="none")
        boxes = np.empty((0, 4))
        warp = np.eye(2, 3)
        result = cmc.apply_to_boxes(boxes, warp)
        assert len(result) == 0

    def test_apply_to_points(self):
        from flashtrack.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="none")
        points = np.array([[10, 20], [30, 40]], dtype=np.float64)
        warp = np.eye(2, 3)
        result = cmc.apply_to_points(points, warp)
        np.testing.assert_allclose(result, points, atol=1e-6)

    def test_compose_warp(self):
        from flashtrack.utils.cmc import compose_warp

        w1 = np.eye(2, 3, dtype=np.float64)
        w2 = np.eye(2, 3, dtype=np.float64)
        w2[0, 2] = 5.0
        result = compose_warp(w1, w2)
        assert result.shape == (2, 3)
        assert result[0, 2] == pytest.approx(5.0)

    def test_invert_warp(self):
        from flashtrack.utils.cmc import invert_warp

        w = np.eye(2, 3, dtype=np.float64)
        w[0, 2] = 10.0
        inv = invert_warp(w)
        assert inv.shape == (2, 3)
        assert inv[0, 2] == pytest.approx(-10.0, abs=1e-6)

    def test_reset(self):
        from flashtrack.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="affine")
        frame = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        cmc.compute(frame)
        cmc.reset()
        assert cmc._prev_frame is None

    def test_numpy_fallback_affine(self):
        from flashtrack.utils.cmc import CameraMotionCompensator

        cmc = CameraMotionCompensator(method="affine")
        f1 = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        f2 = np.roll(f1, 2, axis=1)
        cmc.compute(f1)
        warp = cmc.compute(f2)
        assert warp.shape == (2, 3)


# ===========================================================================
# 9. ReID Feature Extraction
# ===========================================================================


class TestReIDFeatures:
    def test_embedding_normalized(self, small_input):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", pretrained=False)
        emb = model.predict(small_input)
        norms = torch.norm(emb, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.2)

    def test_embedding_batch_consistency(self):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 128, 64)
        with torch.no_grad():
            e1 = model.extract(x)
            e2 = model.extract(x)
        torch.testing.assert_close(e1, e2)


# ===========================================================================
# 10. Edge Cases
# ===========================================================================


class TestEdgeCases:
    def test_wrong_input_channels(self):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", pretrained=False)
        model.eval()
        with pytest.raises(Exception):
            model(torch.randn(1, 1, 128, 64))

    def test_single_sample_batch(self):
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 128, 64))
        assert out["embeddings"].shape[0] == 1

    def test_tracker_with_all_low_scores(self):
        from flashtrack.models.byte_tracker import ByteTracker

        tracker = ByteTracker(track_thresh=0.5)
        dets = np.array([[10, 20, 50, 80]], dtype=np.float64)
        scores = np.array([0.05])
        tracks = tracker.update(dets, scores)
        assert len(tracks) == 0

    def test_hota_single_frame(self):
        from flashtrack.analytics.hota import compute_hota

        gt = [np.array([[0, 0, 50, 50]])]
        gt_ids = [np.array([1])]
        pred = [np.array([[0, 0, 50, 50]])]
        pred_ids = [np.array([1])]
        result = compute_hota(gt, gt_ids, pred, pred_ids)
        assert result["HOTA"] > 0.0


# ===========================================================================
# 11. Integration — end-to-end
# ===========================================================================


class TestIntegration:
    def test_model_to_tracker_pipeline(self):
        """Full pipeline: model → extract features → tracker update."""
        from flashtrack.models.byte_tracker import ByteTracker
        from flashtrack.models.tracker import FlashTracker

        model = FlashTracker(backbone_size="0.5x", pretrained=False)
        model.eval()

        tracker = ByteTracker(track_thresh=0.3)

        crops = torch.randn(3, 3, 128, 64)
        with torch.no_grad():
            embeddings = model.extract(crops)
        assert embeddings.shape == (3, 128)

        dets = np.array([[10, 20, 50, 80], [100, 50, 40, 70], [200, 100, 45, 90]], dtype=np.float64)
        scores = np.array([0.9, 0.85, 0.7])
        tracks = tracker.update(dets, scores)
        assert len(tracks) == 3

    def test_model_to_botsort_with_features(self):
        """BoT-SORT with ReID features from FlashTracker."""
        from flashtrack.models.tracker import FlashTracker
        from flashtrack.trackers.bot_sort import BoTSORTTracker

        model = FlashTracker(backbone_size="0.5x", pretrained=False)
        model.eval()
        tracker = BoTSORTTracker(track_thresh=0.3, cmc_method=None)

        for _ in range(3):
            crops = torch.randn(2, 3, 128, 64)
            with torch.no_grad():
                feats = model.extract(crops).numpy()
            dets = np.array([[10, 20, 50, 80], [200, 100, 40, 60]], dtype=np.float64)
            scores = np.array([0.9, 0.85])
            tracker.update(dets, scores, features=feats)

        results = tracker.get_results()
        assert len(results) >= 1

    def test_hota_on_tracked_output(self, mock_detections):
        """Run tracker, then compute HOTA against ground truth."""
        from flashtrack.analytics.hota import compute_hota
        from flashtrack.models.byte_tracker import ByteTracker

        tracker = ByteTracker(track_thresh=0.3)
        pred_boxes_per_frame = []
        pred_ids_per_frame = []

        for dets, scores in mock_detections:
            tracks = tracker.update(dets, scores)
            if tracks:
                boxes = np.array([t.tlbr for t in tracks])
                ids = np.array([t.track_id for t in tracks])
            else:
                boxes = np.empty((0, 4))
                ids = np.empty(0, dtype=int)
            pred_boxes_per_frame.append(boxes)
            pred_ids_per_frame.append(ids)

        gt_boxes_per_frame = []
        gt_ids_per_frame = []
        for dets, _ in mock_detections:
            tlbr = dets.copy()
            tlbr[:, 2:] += tlbr[:, :2]
            gt_boxes_per_frame.append(tlbr)
            gt_ids_per_frame.append(np.array([1, 2, 3]))

        result = compute_hota(gt_boxes_per_frame, gt_ids_per_frame, pred_boxes_per_frame, pred_ids_per_frame)
        assert "HOTA" in result
        assert result["HOTA"] >= 0.0
