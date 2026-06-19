"""DeepSORTTracker — SORT + ReID appearance matching.

Extends the SORT algorithm (Bewley et al., 2016) with a deep appearance
descriptor (Wojke et al., 2017).  Association is performed using a weighted
combination of Mahalanobis (motion) distance and cosine (appearance) distance.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from flashtrack.utils.kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)


class DeepTrack:
    """Single tracked object with appearance features."""

    _count = 0

    def __init__(
        self,
        bbox_tlwh: np.ndarray,
        score: float,
        feature: Optional[np.ndarray] = None,
        class_id: int = 0,
        n_features: int = 100,
    ):
        self.kalman = KalmanFilter()
        xyah = self._tlwh_to_xyah(bbox_tlwh)
        self.mean, self.covariance = self.kalman.initiate(xyah)

        DeepTrack._count += 1
        self.track_id = DeepTrack._count
        self.score = score
        self.class_id = class_id

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.is_confirmed = False

        self._n_features = n_features
        self.features: List[np.ndarray] = []
        if feature is not None:
            self.features.append(feature / max(np.linalg.norm(feature), 1e-6))

    @staticmethod
    def reset_id():
        DeepTrack._count = 0

    def predict(self):
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_tlwh: np.ndarray, score: float, feature: Optional[np.ndarray] = None):
        xyah = self._tlwh_to_xyah(bbox_tlwh)
        self.mean, self.covariance = self.kalman.update(self.mean, self.covariance, xyah)
        self.hits += 1
        self.time_since_update = 0
        self.score = score

        if feature is not None:
            feat_norm = feature / max(np.linalg.norm(feature), 1e-6)
            self.features.append(feat_norm)
            if len(self.features) > self._n_features:
                self.features.pop(0)

    @property
    def tlwh(self) -> np.ndarray:
        cx, cy, ar, h = self.mean[:4]
        w = ar * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def _tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @property
    def smooth_feature(self) -> Optional[np.ndarray]:
        """Exponentially smoothed appearance feature."""
        if not self.features:
            return None
        feat = np.mean(self.features, axis=0)
        return feat / max(np.linalg.norm(feat), 1e-6)


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


def _cosine_distance(track_features: np.ndarray, det_features: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance between track and detection features.

    Args:
        track_features: [M, D] normalised track features.
        det_features: [N, D] normalised detection features.

    Returns:
        [M, N] cosine distance matrix (1 - cosine_similarity).
    """
    if len(track_features) == 0 or len(det_features) == 0:
        return np.empty((len(track_features), len(det_features)), dtype=np.float64)
    similarity = track_features @ det_features.T
    return 1.0 - np.clip(similarity, -1.0, 1.0)


class DeepSORTTracker:
    """Deep SORT tracker with ReID appearance matching.

    Association cost = lambda_iou * (1 - IoU) + lambda_app * cosine_distance.

    Args:
        max_age: Frames to keep a lost track before deletion.
        n_init: Consecutive detections to confirm a track.
        max_iou_distance: IoU cost gate.
        max_cosine_distance: Cosine distance gate for appearance matching.
        lambda_iou: Weight for IoU component.
        lambda_app: Weight for appearance component.
        n_features: Max stored features per track.
    """

    def __init__(
        self,
        max_age: int = 70,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        lambda_iou: float = 0.5,
        lambda_app: float = 0.5,
        n_features: int = 100,
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.lambda_iou = lambda_iou
        self.lambda_app = lambda_app
        self.n_features = n_features

        self.tracks: List[DeepTrack] = []
        self.frame_id = 0
        DeepTrack.reset_id()

    def reset(self):
        self.tracks = []
        self.frame_id = 0
        DeepTrack.reset_id()

    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Process one frame with detections and optional ReID features.

        Args:
            detections: [N, 4] in tlwh format.
            scores: [N] confidence scores.
            features: [N, D] ReID embeddings (optional, but strongly recommended).
            class_ids: [N] class labels (optional).

        Returns:
            List of dicts with track_id, tlwh, tlbr, score, class_id.
        """
        self.frame_id += 1
        n_dets = len(detections)
        if class_ids is None:
            class_ids = np.zeros(n_dets, dtype=int)
        if features is None:
            features = [None] * n_dets

        # Predict existing tracks
        for t in self.tracks:
            t.predict()

        # Separate confirmed vs unconfirmed tracks
        confirmed = [t for t in self.tracks if t.is_confirmed]
        unconfirmed = [t for t in self.tracks if not t.is_confirmed]

        # ─── Match confirmed tracks using appearance + IoU ───
        matches_c, u_tracks_c, u_dets_c = self._match(
            confirmed, detections, scores, features, use_appearance=True
        )

        for t_idx, d_idx in matches_c:
            confirmed[t_idx].update(
                detections[d_idx], scores[d_idx],
                features[d_idx] if features[d_idx] is not None else None,
            )

        # ─── Match unconfirmed tracks using IoU only ───
        remaining_dets = detections[u_dets_c] if len(u_dets_c) > 0 else np.empty((0, 4))
        remaining_scores = scores[u_dets_c] if len(u_dets_c) > 0 else np.array([])
        remaining_feats = [features[i] for i in u_dets_c]
        class_ids[u_dets_c] if len(u_dets_c) > 0 else np.array([], dtype=int)

        matches_u, u_tracks_u, u_dets_u = self._match(
            unconfirmed, remaining_dets, remaining_scores, remaining_feats, use_appearance=False
        )

        for t_idx, d_idx in matches_u:
            unconfirmed[t_idx].update(
                remaining_dets[d_idx], remaining_scores[d_idx],
                remaining_feats[d_idx] if remaining_feats[d_idx] is not None else None,
            )

        # Mark unmatched unconfirmed as removed
        for i in u_tracks_u:
            unconfirmed[i].time_since_update = self.max_age + 1

        # Create new tracks from remaining unmatched detections
        for i in u_dets_u:
            actual_idx = u_dets_c[i] if len(u_dets_c) > 0 else i
            feat = features[actual_idx] if features[actual_idx] is not None else None
            new_track = DeepTrack(
                detections[actual_idx], scores[actual_idx],
                feature=feat, class_id=class_ids[actual_idx],
                n_features=self.n_features,
            )
            self.tracks.append(new_track)

        # Mark lost confirmed tracks
        for i in u_tracks_c:
            pass  # time_since_update already incremented in predict()

        # Update confirmation status and prune dead tracks
        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= self.n_init:
                t.is_confirmed = True

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks with recent updates
        results = []
        for t in self.tracks:
            if t.is_confirmed and t.time_since_update == 0:
                results.append({
                    "track_id": t.track_id,
                    "tlwh": t.tlwh.tolist(),
                    "tlbr": t.tlbr.tolist(),
                    "score": t.score,
                    "class_id": t.class_id,
                })
        return results

    def _match(
        self,
        tracks: List[DeepTrack],
        detections: np.ndarray,
        scores: np.ndarray,
        features: list,
        use_appearance: bool,
    ):
        if len(tracks) == 0 or len(detections) == 0:
            return (
                np.empty((0, 2), dtype=int),
                list(range(len(tracks))),
                list(range(len(detections))),
            )

        # IoU cost
        track_tlbrs = np.array([t.tlbr for t in tracks])
        det_tlbrs = np.zeros((len(detections), 4))
        for i, d in enumerate(detections):
            det_tlbrs[i] = [d[0], d[1], d[0] + d[2], d[1] + d[3]]

        iou_cost = 1.0 - _iou_batch(track_tlbrs, det_tlbrs)

        if use_appearance and any(t.smooth_feature is not None for t in tracks):
            track_feats = []
            valid_track_mask = []
            for t in tracks:
                sf = t.smooth_feature
                if sf is not None:
                    track_feats.append(sf)
                    valid_track_mask.append(True)
                else:
                    track_feats.append(np.zeros_like(features[0]) if features[0] is not None else np.zeros(128))
                    valid_track_mask.append(False)

            det_feats = []
            for f in features:
                if f is not None:
                    f_norm = f / max(np.linalg.norm(f), 1e-6)
                    det_feats.append(f_norm)
                else:
                    det_feats.append(np.zeros(len(track_feats[0])))

            track_feats = np.array(track_feats)
            det_feats = np.array(det_feats)
            app_cost = _cosine_distance(track_feats, det_feats)

            # Gate by max cosine distance
            app_cost[app_cost > self.max_cosine_distance] = 1.0

            cost = self.lambda_iou * iou_cost + self.lambda_app * app_cost
        else:
            cost = iou_cost

        # Gate by max IoU distance
        cost[iou_cost > self.max_iou_distance] = 1e5

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        matches, u_tracks, u_dets = [], [], []

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1e4:
                matches.append([r, c])
            else:
                u_tracks.append(r)
                u_dets.append(c)

        u_tracks += [i for i in range(len(tracks)) if i not in row_ind]
        u_dets += [j for j in range(len(detections)) if j not in col_ind]

        matches = np.array(matches).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int)
        return matches, u_tracks, u_dets
