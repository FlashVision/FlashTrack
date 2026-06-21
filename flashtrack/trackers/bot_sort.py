"""BoT-SORT — robust multi-object tracker with camera motion compensation.

Implements BoT-SORT (Aharon et al., 2022):
  - Camera motion compensation via affine/homography estimation.
  - Improved Kalman filter with Noise Scale Adaptive (NSA) update.
  - Optional appearance-feature integration for re-identification.

References:
    Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking", 2022.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from flashtrack.utils.kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)


class _TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class BoTTrack:
    """Single tracked object managed by BoT-SORT."""

    _count = 0

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        class_id: int = 0,
        feature: Optional[np.ndarray] = None,
    ):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = score
        self.class_id = class_id

        self.kalman = KalmanFilter()
        self.mean, self.covariance = None, None

        self.is_activated = False
        self.state = _TrackState.NEW
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

        self.smooth_feat = None
        self.alpha = 0.9
        if feature is not None:
            self.smooth_feat = feature / max(np.linalg.norm(feature), 1e-6)
        self._features: List[np.ndarray] = []
        if feature is not None:
            self._features.append(self.smooth_feat.copy())

    @staticmethod
    def next_id() -> int:
        BoTTrack._count += 1
        return BoTTrack._count

    @staticmethod
    def reset_id():
        BoTTrack._count = 0

    def activate(self, frame_id: int):
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman.initiate(self._tlwh_to_xyah())
        self.state = _TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0

    def re_activate(self, new_track: BoTTrack, frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman.update(
            self.mean, self.covariance, new_track._tlwh_to_xyah()
        )
        self.tracklet_len = 0
        self.state = _TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.tlwh = new_track.tlwh
        self._update_feature(new_track.smooth_feat)
        if new_id:
            self.track_id = self.next_id()

    def predict(self):
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)

    def update(self, new_track: BoTTrack, frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.tlwh = new_track.tlwh
        self.score = new_track.score

        measurement = new_track._tlwh_to_xyah()
        self.mean, self.covariance = self._nsa_kalman_update(measurement, new_track.score)
        self.state = _TrackState.TRACKED
        self.is_activated = True
        self._update_feature(new_track.smooth_feat)

    def _nsa_kalman_update(
        self,
        measurement: np.ndarray,
        detection_score: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Noise Scale Adaptive (NSA) Kalman update.

        Scales the measurement noise inversely with detection confidence.
        """
        proj_mean, proj_cov = self.kalman.project(self.mean, self.covariance)
        nsa_factor = 1.0 / max(detection_score, 0.1)
        proj_cov = proj_cov * nsa_factor

        import scipy.linalg
        chol = scipy.linalg.cho_factor(proj_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            chol,
            (self.covariance @ self.kalman._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - proj_mean
        new_mean = self.mean + innovation @ kalman_gain.T
        new_cov = self.covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov

    def _update_feature(self, feat: Optional[np.ndarray]):
        if feat is None:
            return
        feat = feat / max(np.linalg.norm(feat), 1e-6)
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat /= max(np.linalg.norm(self.smooth_feat), 1e-6)
        self._features.append(feat)
        if len(self._features) > 100:
            self._features.pop(0)

    def mark_lost(self):
        self.state = _TrackState.LOST

    def mark_removed(self):
        self.state = _TrackState.REMOVED

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def _tlwh_to_xyah(self) -> np.ndarray:
        ret = np.asarray(self.tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @property
    def predicted_tlwh(self) -> np.ndarray:
        if self.mean is None:
            return self.tlwh.copy()
        cx, cy, ar, h = self.mean[:4]
        w = ar * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    @property
    def predicted_tlbr(self) -> np.ndarray:
        tlwh = self.predicted_tlwh
        tlwh[2:] += tlwh[:2]
        return tlwh

    def apply_affine(self, warp_matrix: np.ndarray):
        """Apply camera motion compensation affine transform to Kalman state."""
        if self.mean is None:
            return
        cx, cy = self.mean[0], self.mean[1]
        pt = np.array([cx, cy, 1.0])
        new_pt = warp_matrix @ pt
        self.mean[0] = new_pt[0]
        self.mean[1] = new_pt[1]


def _iou_batch(a_tlbr: np.ndarray, b_tlbr: np.ndarray) -> np.ndarray:
    if len(a_tlbr) == 0 or len(b_tlbr) == 0:
        return np.empty((len(a_tlbr), len(b_tlbr)), dtype=np.float64)
    a = np.asarray(a_tlbr, dtype=np.float64)
    b = np.asarray(b_tlbr, dtype=np.float64)
    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)), dtype=np.float64)
    return 1.0 - np.clip(a @ b.T, -1.0, 1.0)


def _linear_assignment(cost_matrix: np.ndarray, thresh: float):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, u_a, u_b = [], [], []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            u_a.append(r)
            u_b.append(c)
        else:
            matches.append([r, c])
    u_a += [i for i in range(cost_matrix.shape[0]) if i not in row_ind]
    u_b += [j for j in range(cost_matrix.shape[1]) if j not in col_ind]
    return (
        np.array(matches).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int),
        u_a, u_b,
    )


class BoTSORTTracker:
    """BoT-SORT multi-object tracker.

    Args:
        track_thresh: High-confidence detection threshold.
        track_buffer: Frames to keep lost tracks alive.
        match_thresh: IoU threshold for matching.
        low_thresh: Lower-bound confidence for second association.
        frame_rate: Video frame rate.
        lambda_iou: IoU cost weight.
        lambda_app: Appearance cost weight.
        cmc_method: Camera motion compensation method (``"affine"`` or ``None``).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        low_thresh: float = 0.1,
        frame_rate: int = 30,
        lambda_iou: float = 0.5,
        lambda_app: float = 0.5,
        cmc_method: Optional[str] = "affine",
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.lambda_iou = lambda_iou
        self.lambda_app = lambda_app
        self.cmc_method = cmc_method

        self.tracked_stracks: List[BoTTrack] = []
        self.lost_stracks: List[BoTTrack] = []
        self.removed_stracks: List[BoTTrack] = []
        self.frame_id = 0
        self._prev_frame_gray: Optional[np.ndarray] = None

        BoTTrack.reset_id()

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self._prev_frame_gray = None
        BoTTrack.reset_id()

    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        frame_gray: Optional[np.ndarray] = None,
    ) -> List[BoTTrack]:
        """Process one frame.

        Args:
            detections: [N, 4] bounding boxes in tlwh format.
            scores: [N] confidence scores.
            class_ids: [N] class IDs (optional).
            features: [N, D] appearance features (optional).
            frame_gray: Grayscale frame for camera motion compensation.

        Returns:
            List of active ``BoTTrack`` objects.
        """
        self.frame_id += 1
        if class_ids is None:
            class_ids = np.zeros(len(scores), dtype=int)
        if features is None:
            features = [None] * len(scores)

        # Camera motion compensation
        if self.cmc_method and frame_gray is not None and self._prev_frame_gray is not None:
            warp = self._estimate_affine(self._prev_frame_gray, frame_gray)
            if warp is not None:
                for t in self.tracked_stracks + self.lost_stracks:
                    t.apply_affine(warp)
        self._prev_frame_gray = frame_gray

        activated, refind, lost, removed = [], [], [], []

        # Split detections
        high_mask = scores >= self.track_thresh
        low_mask = (scores >= self.low_thresh) & ~high_mask

        high_stracks = [
            BoTTrack(d, s, c, f)
            for d, s, c, f in zip(
                detections[high_mask], scores[high_mask],
                class_ids[high_mask],
                [features[i] for i in np.where(high_mask)[0]],
            )
        ]

        # Predict existing tracks
        pool = self.tracked_stracks + self.lost_stracks
        for t in pool:
            t.predict()

        # First association — combined IoU + appearance
        track_tlbrs = np.array([t.predicted_tlbr for t in pool]) if pool else np.empty((0, 4))
        det_tlbrs = np.array([d.tlbr for d in high_stracks]) if high_stracks else np.empty((0, 4))

        iou_cost = 1.0 - _iou_batch(track_tlbrs, det_tlbrs)

        use_app = (
            self.lambda_app > 0
            and any(t.smooth_feat is not None for t in pool)
            and any(d.smooth_feat is not None for d in high_stracks)
        )

        if use_app:
            t_feats = np.array([
                t.smooth_feat if t.smooth_feat is not None else np.zeros(128)
                for t in pool
            ])
            d_feats = np.array([
                d.smooth_feat if d.smooth_feat is not None else np.zeros(128)
                for d in high_stracks
            ])
            app_cost = _cosine_distance(t_feats, d_feats)
            cost = self.lambda_iou * iou_cost + self.lambda_app * app_cost
        else:
            cost = iou_cost

        matches, u_tracks, u_dets = _linear_assignment(cost, 1.0 - self.match_thresh)

        for t_idx, d_idx in matches:
            track = pool[t_idx]
            det = high_stracks[d_idx]
            if track.state == _TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)

        # Second association — low-confidence + IoU only
        r_tracked = [pool[i] for i in u_tracks if pool[i].state == _TrackState.TRACKED]

        low_dets = detections[low_mask]
        low_scores = scores[low_mask]
        low_cls = class_ids[low_mask]
        low_stracks = [BoTTrack(d, s, c) for d, s, c in zip(low_dets, low_scores, low_cls)]

        r_tlbrs = np.array([t.predicted_tlbr for t in r_tracked]) if r_tracked else np.empty((0, 4))
        l_tlbrs = np.array([d.tlbr for d in low_stracks]) if low_stracks else np.empty((0, 4))

        iou2 = _iou_batch(r_tlbrs, l_tlbrs)
        cost2 = 1.0 - iou2
        m2, u2, _ = _linear_assignment(cost2, 0.5)

        for t_idx, d_idx in m2:
            track = r_tracked[t_idx]
            det = low_stracks[d_idx]
            if track.state == _TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)

        for i in u2:
            track = r_tracked[i]
            if track.state != _TrackState.LOST:
                track.mark_lost()
                lost.append(track)

        # New tracks
        for i in u_dets:
            det = high_stracks[i]
            if det.score >= self.track_thresh:
                det.activate(self.frame_id)
                activated.append(det)

        # Expire old lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        # Merge lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == _TrackState.TRACKED]
        self.tracked_stracks = _merge(self.tracked_stracks, activated)
        self.tracked_stracks = _merge(self.tracked_stracks, refind)
        self.lost_stracks = _subtract(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = _subtract(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)

        return [t for t in self.tracked_stracks if t.is_activated]

    @staticmethod
    def _estimate_affine(
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Estimate 2x3 affine warp between consecutive frames.

        Uses sparse optical flow with RANSAC for robustness.
        """
        try:
            import cv2
        except ImportError:
            return None

        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01,
            minDistance=30, blockSize=3,
        )
        if prev_pts is None or len(prev_pts) < 4:
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None,
        )
        if curr_pts is None:
            return None

        mask = status.flatten() == 1
        prev_pts = prev_pts[mask].reshape(-1, 2)
        curr_pts = curr_pts[mask].reshape(-1, 2)

        if len(prev_pts) < 4:
            return None

        warp, inliers = cv2.estimateAffinePartial2D(
            prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0,
        )
        return warp

    def get_results(self) -> List[Dict]:
        results = []
        for t in self.tracked_stracks:
            if t.is_activated:
                results.append({
                    "track_id": t.track_id,
                    "tlwh": t.tlwh.tolist(),
                    "tlbr": t.tlbr.tolist(),
                    "score": t.score,
                    "class_id": t.class_id,
                })
        return results


def _merge(a: List[BoTTrack], b: List[BoTTrack]) -> List[BoTTrack]:
    ids = {t.track_id for t in a}
    result = list(a)
    for t in b:
        if t.track_id not in ids:
            result.append(t)
            ids.add(t.track_id)
    return result


def _subtract(a: List[BoTTrack], b: List[BoTTrack]) -> List[BoTTrack]:
    ids_b = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids_b]
