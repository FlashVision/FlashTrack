"""HOTA — Higher Order Tracking Accuracy metric.

Implements HOTA with its decomposition into Detection Accuracy (DetA),
Association Accuracy (AssA), and Localisation Accuracy (LocA).

References:
    Luiten et al., "HOTA: A Higher Order Metric for Evaluating
    Multi-Object Tracking", IJCV 2021.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def compute_iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """Compute IoU matrix between two sets of [x1, y1, x2, y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float64)

    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)

    x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _match_at_threshold(
    iou_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match GT and predictions at a given IoU threshold using Hungarian."""
    if iou_matrix.size == 0:
        return (
            [],
            list(range(iou_matrix.shape[0])),
            list(range(iou_matrix.shape[1])),
        )

    cost = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    unmatched_gt = []
    unmatched_pred = []

    matched_rows = set()
    matched_cols = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= threshold:
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
        else:
            unmatched_gt.append(r)
            unmatched_pred.append(c)

    unmatched_gt += [i for i in range(iou_matrix.shape[0]) if i not in matched_rows and i not in unmatched_gt]
    unmatched_pred += [j for j in range(iou_matrix.shape[1]) if j not in matched_cols and j not in unmatched_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_hota(
    gt_boxes_per_frame: List[np.ndarray],
    gt_ids_per_frame: List[np.ndarray],
    pred_boxes_per_frame: List[np.ndarray],
    pred_ids_per_frame: List[np.ndarray],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute HOTA and its components (DetA, AssA, LocA).

    Args:
        gt_boxes_per_frame: List of [M, 4] GT boxes per frame (x1, y1, x2, y2).
        gt_ids_per_frame: List of [M] GT identity IDs per frame.
        pred_boxes_per_frame: List of [N, 4] predicted boxes per frame.
        pred_ids_per_frame: List of [N] predicted track IDs per frame.
        iou_thresholds: IoU thresholds (default: 0.05 to 0.95 in 0.05 steps).

    Returns:
        Dict with ``'HOTA'``, ``'DetA'``, ``'AssA'``, ``'LocA'``, and
        per-threshold values.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.05 + 0.05 * i, 2) for i in range(19)]

    num_frames = len(gt_boxes_per_frame)

    hota_per_thresh = {}

    for alpha in iou_thresholds:
        # Per-frame matching at this threshold
        tp_total = 0
        fn_total = 0
        fp_total = 0
        iou_sum = 0.0

        # For association accuracy: track gt_id -> pred_id correspondences
        gt_to_pred: Dict[int, List[int]] = {}
        pred_to_gt: Dict[int, List[int]] = {}

        for frame_idx in range(num_frames):
            gt_boxes = gt_boxes_per_frame[frame_idx]
            gt_ids = gt_ids_per_frame[frame_idx]
            pred_boxes = pred_boxes_per_frame[frame_idx]
            pred_ids = pred_ids_per_frame[frame_idx]

            n_gt = len(gt_boxes)
            n_pred = len(pred_boxes)

            if n_gt == 0:
                fp_total += n_pred
                continue
            if n_pred == 0:
                fn_total += n_gt
                continue

            iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
            matches, u_gt, u_pred = _match_at_threshold(iou_mat, alpha)

            tp_total += len(matches)
            fn_total += len(u_gt)
            fp_total += len(u_pred)

            for gi, pi in matches:
                iou_sum += iou_mat[gi, pi]
                g_id = int(gt_ids[gi])
                p_id = int(pred_ids[pi])
                gt_to_pred.setdefault(g_id, []).append(p_id)
                pred_to_gt.setdefault(p_id, []).append(g_id)

        # Detection accuracy
        det_a = tp_total / max(tp_total + fn_total + fp_total, 1)

        # Localisation accuracy (average IoU of TPs)
        loc_a = iou_sum / max(tp_total, 1)

        # Association accuracy
        ass_a = _compute_association_accuracy(gt_to_pred, pred_to_gt)

        # HOTA = sqrt(DetA * AssA)
        hota = math.sqrt(det_a * ass_a) if det_a > 0 and ass_a > 0 else 0.0

        hota_per_thresh[alpha] = {
            "HOTA": hota,
            "DetA": det_a,
            "AssA": ass_a,
            "LocA": loc_a,
        }

    # Average over thresholds
    avg_hota = np.mean([v["HOTA"] for v in hota_per_thresh.values()])
    avg_det_a = np.mean([v["DetA"] for v in hota_per_thresh.values()])
    avg_ass_a = np.mean([v["AssA"] for v in hota_per_thresh.values()])
    avg_loc_a = np.mean([v["LocA"] for v in hota_per_thresh.values()])

    return {
        "HOTA": float(avg_hota),
        "DetA": float(avg_det_a),
        "AssA": float(avg_ass_a),
        "LocA": float(avg_loc_a),
        "per_threshold": hota_per_thresh,
    }


import math


def _compute_association_accuracy(
    gt_to_pred: Dict[int, List[int]],
    pred_to_gt: Dict[int, List[int]],
) -> float:
    """Compute Association Accuracy (AssA).

    For each matched (gt_id, pred_id) pair, compute the Jaccard index of
    their temporal association: |TPA| / (|TPA| + |FPA| + |FNA|).

    TPA(c): frames where gt_id c is matched to pred_id k.
    FPA(c): frames where pred_id k is matched to some other gt_id.
    FNA(c): frames where gt_id c is matched to some other pred_id.
    """
    if not gt_to_pred:
        return 0.0

    all_gt_ids = set(gt_to_pred.keys())
    all_pred_ids = set(pred_to_gt.keys())

    ass_scores = []

    for g_id in all_gt_ids:
        pred_matches = gt_to_pred[g_id]

        # For each unique pred_id matched to this gt_id
        pred_counter: Dict[int, int] = {}
        for p_id in pred_matches:
            pred_counter[p_id] = pred_counter.get(p_id, 0) + 1

        for p_id, tpa in pred_counter.items():
            fna = len(pred_matches) - tpa
            fpa = len(pred_to_gt.get(p_id, [])) - tpa

            ass_score = tpa / max(tpa + fna + fpa, 1)
            ass_scores.extend([ass_score] * tpa)

    return float(np.mean(ass_scores)) if ass_scores else 0.0


def compute_hota_summary(
    gt_boxes_per_frame: List[np.ndarray],
    gt_ids_per_frame: List[np.ndarray],
    pred_boxes_per_frame: List[np.ndarray],
    pred_ids_per_frame: List[np.ndarray],
) -> str:
    """Compute HOTA and return a formatted summary string."""
    metrics = compute_hota(
        gt_boxes_per_frame, gt_ids_per_frame,
        pred_boxes_per_frame, pred_ids_per_frame,
    )
    lines = [
        f"HOTA:  {metrics['HOTA']:.4f}",
        f"DetA:  {metrics['DetA']:.4f}",
        f"AssA:  {metrics['AssA']:.4f}",
        f"LocA:  {metrics['LocA']:.4f}",
    ]
    return "\n".join(lines)
