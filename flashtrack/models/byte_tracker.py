"""ByteTracker — two-stage IoU-based multi-object tracker.

Implements ByteTrack (Zhang et al., 2022):
  1. First association: high-confidence detections ↔ existing tracks via IoU.
  2. Second association: remaining low-confidence detections ↔ unmatched tracks.
  3. Track lifecycle management (init, activate, re-activate, delete).

Uses a Kalman filter for motion prediction and the Hungarian algorithm for
optimal bipartite matching.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from flashtrack.utils.kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)


class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


class STrack:
    """Single tracked object managed by ByteTracker."""

    _count = 0

    def __init__(self, tlwh: np.ndarray, score: float, class_id: int = 0):
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = score
        self.class_id = class_id

        self.kalman = KalmanFilter()
        self.mean, self.covariance = None, None

        self.is_activated = False
        self.state = TrackState.NEW
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    @staticmethod
    def next_id() -> int:
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id():
        STrack._count = 0

    def activate(self, frame_id: int):
        """Start a new track."""
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman.initiate(self._tlwh_to_xyah())
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Re-activate a lost track with a new detection."""
        self.mean, self.covariance = self.kalman.update(
            self.mean, self.covariance, new_track._tlwh_to_xyah()
        )
        self.tracklet_len = 0
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.tlwh = new_track.tlwh
        if new_id:
            self.track_id = self.next_id()

    def predict(self):
        """Propagate state forward using the Kalman filter."""
        self.mean, self.covariance = self.kalman.predict(self.mean, self.covariance)

    def update(self, new_track: STrack, frame_id: int):
        """Update track with a matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman.update(
            self.mean, self.covariance, new_track._tlwh_to_xyah()
        )
        self.state = TrackState.TRACKED
        self.is_activated = True
        self.score = new_track.score
        self.tlwh = new_track.tlwh

    def mark_lost(self):
        self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED

    @property
    def tlbr(self) -> np.ndarray:
        """Convert (top-left-w-h) to (top-left-bottom-right)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def _tlwh_to_xyah(self) -> np.ndarray:
        """Convert tlwh to (center_x, center_y, aspect_ratio, height)."""
        ret = np.asarray(self.tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @property
    def predicted_tlwh(self) -> np.ndarray:
        """Get predicted tlwh from Kalman state."""
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


def _iou_batch(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes in tlbr format."""
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.empty((len(atlbrs), len(btlbrs)), dtype=np.float64)

    a = np.asarray(atlbrs, dtype=np.float64)
    b = np.asarray(btlbrs, dtype=np.float64)

    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Solve linear assignment and filter by cost threshold."""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_a, unmatched_b = [], [], []

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            unmatched_a.append(r)
            unmatched_b.append(c)
        else:
            matches.append([r, c])

    unmatched_a += [i for i in range(cost_matrix.shape[0]) if i not in row_ind]
    unmatched_b += [j for j in range(cost_matrix.shape[1]) if j not in col_ind]

    return np.array(matches).reshape(-1, 2) if matches else np.empty((0, 2), dtype=int), unmatched_a, unmatched_b


class ByteTracker:
    """ByteTrack multi-object tracker.

    Args:
        track_thresh: Detection confidence threshold for first association.
        track_buffer: Max frames to keep a lost track alive.
        match_thresh: IoU threshold for matching.
        low_thresh: Lower bound for second-stage association (default 0.1).
        frame_rate: Video frame rate (adjusts track_buffer lifetime).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        low_thresh: float = 0.1,
        frame_rate: int = 30,
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.low_thresh = low_thresh
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

        STrack.reset_id()

    def reset(self):
        """Reset tracker state."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.reset_id()

    def update(
        self,
        detections: np.ndarray,
        scores: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
    ) -> List[STrack]:
        """Process one frame of detections.

        Args:
            detections: [N, 4] bounding boxes in tlwh format.
            scores: [N] confidence scores.
            class_ids: [N] class IDs (optional).

        Returns:
            List of active STrack objects with updated state.
        """
        self.frame_id += 1
        if class_ids is None:
            class_ids = np.zeros(len(scores), dtype=int)

        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Split detections by confidence
        high_mask = scores >= self.track_thresh
        low_mask = (scores >= self.low_thresh) & ~high_mask

        high_dets = detections[high_mask]
        high_scores = scores[high_mask]
        high_cls = class_ids[high_mask]

        low_dets = detections[low_mask]
        low_scores = scores[low_mask]
        low_cls = class_ids[low_mask]

        # Create STrack objects for high-confidence detections
        high_stracks = [STrack(d, s, c) for d, s, c in zip(high_dets, high_scores, high_cls)]

        # Predict existing tracks
        strack_pool = self.tracked_stracks + self.lost_stracks
        for t in strack_pool:
            t.predict()

        # ─── First association: high-conf dets ↔ tracks ───
        track_tlbrs = np.array([t.predicted_tlbr for t in strack_pool]) if strack_pool else np.empty((0, 4))
        det_tlbrs = np.array([d.tlbr for d in high_stracks]) if high_stracks else np.empty((0, 4))

        iou_matrix = _iou_batch(track_tlbrs, det_tlbrs)
        cost_matrix = 1.0 - iou_matrix

        matches, u_tracks, u_dets = _linear_assignment(cost_matrix, 1.0 - self.match_thresh)

        for t_idx, d_idx in matches:
            track = strack_pool[t_idx]
            det = high_stracks[d_idx]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        # ─── Second association: low-conf dets ↔ unmatched tracked ───
        r_tracked = [strack_pool[i] for i in u_tracks if strack_pool[i].state == TrackState.TRACKED]

        low_stracks = [STrack(d, s, c) for d, s, c in zip(low_dets, low_scores, low_cls)]

        r_tlbrs = np.array([t.predicted_tlbr for t in r_tracked]) if r_tracked else np.empty((0, 4))
        low_tlbrs = np.array([d.tlbr for d in low_stracks]) if low_stracks else np.empty((0, 4))

        iou_matrix2 = _iou_batch(r_tlbrs, low_tlbrs)
        cost_matrix2 = 1.0 - iou_matrix2

        matches2, u_tracks2, _ = _linear_assignment(cost_matrix2, 1.0 - 0.5)

        for t_idx, d_idx in matches2:
            track = r_tracked[t_idx]
            det = low_stracks[d_idx]
            if track.state == TrackState.TRACKED:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for i in u_tracks2:
            track = r_tracked[i]
            if not track.state == TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)

        # ─── Initialize new tracks from unmatched high-conf dets ───
        for i in u_dets:
            det = high_stracks[i]
            if det.score >= self.track_thresh:
                det.activate(self.frame_id)
                activated_stracks.append(det)

        # ─── Expire lost tracks ───
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update state lists
        self.tracked_stracks = [
            t for t in self.tracked_stracks
            if t.state == TrackState.TRACKED
        ]
        self.tracked_stracks = _merge_lists(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = _merge_lists(self.tracked_stracks, refind_stracks)
        self.lost_stracks = _subtract_lists(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = _subtract_lists(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        return [t for t in self.tracked_stracks if t.is_activated]

    def get_results(self) -> List[Dict]:
        """Return current active tracks as dicts."""
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


def _merge_lists(a: List[STrack], b: List[STrack]) -> List[STrack]:
    """Merge two track lists, keeping unique IDs."""
    existing = {t.track_id for t in a}
    result = list(a)
    for t in b:
        if t.track_id not in existing:
            result.append(t)
            existing.add(t.track_id)
    return result


def _subtract_lists(a: List[STrack], b: List[STrack]) -> List[STrack]:
    """Remove tracks in b from a."""
    ids_b = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids_b]
