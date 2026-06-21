"""OC-SORT — Observation-Centric SORT.

Implements OC-SORT (Cao et al., CVPR 2023):
  - Observation-Centric Momentum (OCM): uses last observation instead of
    Kalman prediction for unmatched tracks when computing IoU.
  - Observation-Centric Recovery (OCR): virtual trajectory generation for
    tracks that were lost and then re-found.
  - Virtual trajectory interpolation for lost frames.

References:
    Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust
    Multi-Object Tracking", CVPR 2023.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from flashtrack.utils.kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)


class OCTrack:
    """Single tracked object managed by OC-SORT."""

    _count = 0

    def __init__(self, bbox_tlwh: np.ndarray, score: float, class_id: int = 0):
        self.kalman = KalmanFilter()
        xyah = self._tlwh_to_xyah(bbox_tlwh)
        self.mean, self.covariance = self.kalman.initiate(xyah)

        OCTrack._count += 1
        self.track_id = OCTrack._count
        self.score = score
        self.class_id = class_id

        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        self.last_observation = bbox_tlwh.copy()
        self.observations: List[Tuple[int, np.ndarray]] = []
        self.velocity = np.zeros(2)
        self._frozen_tlwh: Optional[np.ndarray] = None

    @staticmethod
    def reset_id():
        OCTrack._count = 0

    def predict(self):
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_tlwh: np.ndarray, score: float, frame_id: int):
        xyah = self._tlwh_to_xyah(bbox_tlwh)
        self.mean, self.covariance = self.kalman.update(self.mean, self.covariance, xyah)
        self.hits += 1
        self.time_since_update = 0
        self.score = score

        old_center = self.last_observation[:2] + self.last_observation[2:] / 2
        new_center = bbox_tlwh[:2] + bbox_tlwh[2:] / 2
        self.velocity = new_center - old_center

        self.last_observation = bbox_tlwh.copy()
        self.observations.append((frame_id, bbox_tlwh.copy()))
        self._frozen_tlwh = None

    def freeze(self):
        """Freeze at last observation (for OCM — use observation, not prediction)."""
        self._frozen_tlwh = self.last_observation.copy()

    @property
    def tlwh(self) -> np.ndarray:
        if self._frozen_tlwh is not None:
            return self._frozen_tlwh
        cx, cy, ar, h = self.mean[:4]
        w = ar * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    @property
    def tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def observation_tlbr(self) -> np.ndarray:
        """Use last observation for OCM-based matching."""
        ret = self.last_observation.copy()
        ret[2:] += ret[:2]
        return ret

    def virtual_trajectory(self, target_frame: int) -> np.ndarray:
        """Generate virtual bounding box at target_frame via linear interpolation.

        Used for OCR: when a lost track is re-found, interpolate the trajectory
        for the missing frames.
        """
        if len(self.observations) < 2:
            return self.last_observation.copy()

        last_frame, last_box = self.observations[-1]
        prev_frame, prev_box = self.observations[-2]

        dt_total = last_frame - prev_frame
        if dt_total <= 0:
            return last_box.copy()

        dt = target_frame - prev_frame
        ratio = dt / dt_total
        return prev_box + (last_box - prev_box) * ratio

    @staticmethod
    def _tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret


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


def _velocity_direction_consistency(
    track: OCTrack,
    detection_tlwh: np.ndarray,
) -> float:
    """Score based on consistency between track velocity and detection direction."""
    if np.linalg.norm(track.velocity) < 1e-6:
        return 1.0

    det_center = detection_tlwh[:2] + detection_tlwh[2:] / 2
    obs_center = track.last_observation[:2] + track.last_observation[2:] / 2
    direction = det_center - obs_center

    if np.linalg.norm(direction) < 1e-6:
        return 1.0

    cos_sim = np.dot(track.velocity, direction) / (
        np.linalg.norm(track.velocity) * np.linalg.norm(direction)
    )
    return float(max(0.0, cos_sim))


class OCSORTTracker:
    """Observation-Centric SORT tracker.

    Args:
        max_age: Maximum frames without update before track deletion.
        min_hits: Minimum consecutive hits to confirm a track.
        iou_threshold: IoU threshold for matching.
        delta_t: Time window for virtual trajectory (OCR).
        use_ocm: Enable observation-centric momentum.
        use_ocr: Enable observation-centric recovery.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        use_ocm: bool = True,
        use_ocr: bool = True,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.use_ocm = use_ocm
        self.use_ocr = use_ocr

        self.tracks: List[OCTrack] = []
        self.frame_id = 0
        OCTrack.reset_id()

    def reset(self):
        self.tracks = []
        self.frame_id = 0
        OCTrack.reset_id()

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

        # OCM: use last observation instead of Kalman prediction for unmatched
        if self.use_ocm:
            for t in self.tracks:
                if t.time_since_update > 0:
                    t.freeze()

        # Build cost matrix
        if self.use_ocm:
            track_boxes = np.array([t.observation_tlbr for t in self.tracks]) if self.tracks else np.empty((0, 4))
        else:
            track_boxes = np.array([t.tlbr for t in self.tracks]) if self.tracks else np.empty((0, 4))

        det_tlbrs = np.zeros((len(detections), 4))
        for i, d in enumerate(detections):
            det_tlbrs[i] = [d[0], d[1], d[0] + d[2], d[1] + d[3]]

        iou_matrix = _iou_batch(track_boxes, det_tlbrs)
        cost_matrix = 1.0 - iou_matrix

        matched_indices = []
        unmatched_dets = list(range(len(detections)))

        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_t = set()
            matched_d = set()
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))
                    matched_t.add(r)
                    matched_d.add(c)
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_t]
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_d]
        else:
            unmatched_tracks = list(range(len(self.tracks)))

        # Update matched tracks
        for t_idx, d_idx in matched_indices:
            self.tracks[t_idx].update(detections[d_idx], scores[d_idx], self.frame_id)

        # OCR: try to recover unmatched tracks using virtual trajectories
        if self.use_ocr and unmatched_tracks and unmatched_dets:
            virtual_boxes = []
            valid_unmatched = []
            for t_idx in unmatched_tracks:
                track = self.tracks[t_idx]
                if track.time_since_update <= self.delta_t and len(track.observations) >= 2:
                    vbox = track.virtual_trajectory(self.frame_id)
                    vbox_tlbr = vbox.copy()
                    vbox_tlbr[2:] += vbox_tlbr[:2]
                    virtual_boxes.append(vbox_tlbr)
                    valid_unmatched.append(t_idx)

            if virtual_boxes and unmatched_dets:
                v_boxes = np.array(virtual_boxes)
                u_det_boxes = det_tlbrs[unmatched_dets]
                v_iou = _iou_batch(v_boxes, u_det_boxes)
                v_cost = 1.0 - v_iou

                if v_cost.size > 0:
                    vr, vc = linear_sum_assignment(v_cost)
                    recovered_dets = set()
                    for r, c in zip(vr, vc):
                        if v_iou[r, c] >= self.iou_threshold:
                            t_idx = valid_unmatched[r]
                            d_idx = unmatched_dets[c]
                            self.tracks[t_idx].update(
                                detections[d_idx], scores[d_idx], self.frame_id
                            )
                            recovered_dets.add(d_idx)
                    unmatched_dets = [d for d in unmatched_dets if d not in recovered_dets]

        # Create new tracks
        for d_idx in unmatched_dets:
            new_track = OCTrack(detections[d_idx], scores[d_idx], class_ids[d_idx])
            new_track.observations.append((self.frame_id, detections[d_idx].copy()))
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
