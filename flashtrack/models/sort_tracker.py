"""SORTTracker — Simple Online and Realtime Tracking.

Implements the SORT algorithm (Bewley et al., 2016):
  - Kalman filter for motion prediction (constant velocity model).
  - Hungarian algorithm for IoU-based assignment.
  - Simple track lifecycle (creation / deletion by age).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from flashtrack.utils.kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)


class KalmanTrack:
    """Single tracked object managed by SORTTracker."""

    _count = 0

    def __init__(self, bbox_tlwh: np.ndarray, score: float, class_id: int = 0):
        self.kalman = KalmanFilter()
        xyah = self._tlwh_to_xyah(bbox_tlwh)
        self.mean, self.covariance = self.kalman.initiate(xyah)

        KalmanTrack._count += 1
        self.track_id = KalmanTrack._count
        self.score = score
        self.class_id = class_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    @staticmethod
    def reset_id():
        KalmanTrack._count = 0

    def predict(self):
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_tlwh: np.ndarray, score: float):
        xyah = self._tlwh_to_xyah(bbox_tlwh)
        self.mean, self.covariance = self.kalman.update(self.mean, self.covariance, xyah)
        self.hits += 1
        self.time_since_update = 0
        self.score = score

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


def _iou_batch(a_tlbr: np.ndarray, b_tlbr: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two sets of tlbr boxes."""
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


class SORTTracker:
    """Simple Online and Realtime Tracker.

    Args:
        max_age: Maximum frames to keep a track without update before deletion.
        min_hits: Minimum consecutive hits to confirm a track.
        iou_threshold: IoU threshold for matching.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanTrack] = []
        self.frame_id = 0
        KalmanTrack.reset_id()

    def reset(self):
        self.tracks = []
        self.frame_id = 0
        KalmanTrack.reset_id()

    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Process one frame.

        Args:
            detections: [N, 4] in tlwh format.
            scores: [N] confidence scores.
            class_ids: [N] class labels (optional).

        Returns:
            List of dicts with track_id, tlwh, tlbr, score, class_id.
        """
        self.frame_id += 1
        if class_ids is None:
            class_ids = np.zeros(len(scores), dtype=int)

        # Predict existing tracks
        for t in self.tracks:
            t.predict()

        # Build cost matrix (1 - IoU)
        track_tlbrs = np.array([t.tlbr for t in self.tracks]) if self.tracks else np.empty((0, 4))
        det_tlbrs = np.zeros((len(detections), 4))
        for i, d in enumerate(detections):
            det_tlbrs[i] = [d[0], d[1], d[0] + d[2], d[1] + d[3]]

        iou_matrix = _iou_batch(track_tlbrs, det_tlbrs)
        cost_matrix = 1.0 - iou_matrix

        # Hungarian matching
        matched_indices = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))

        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))

            matched_t = {m[0] for m in matched_indices}
            matched_d = {m[1] for m in matched_indices}
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_t]
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_d]

        # Update matched tracks
        for t_idx, d_idx in matched_indices:
            self.tracks[t_idx].update(detections[d_idx], scores[d_idx])

        # Create new tracks from unmatched detections
        for d_idx in unmatched_dets:
            new_track = KalmanTrack(detections[d_idx], scores[d_idx], class_ids[d_idx])
            self.tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        results = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                results.append({
                    "track_id": t.track_id,
                    "tlwh": t.tlwh.tolist(),
                    "tlbr": t.tlbr.tolist(),
                    "score": t.score,
                    "class_id": t.class_id,
                })
        return results
