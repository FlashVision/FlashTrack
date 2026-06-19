"""Evaluation metrics for multi-object tracking.

MOTA, MOTP, IDF1, ID switches, and track fragmentation.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between GT and predicted boxes.

    Args:
        gt_boxes: [M, 4] ground-truth boxes (x1, y1, x2, y2).
        pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2).

    Returns:
        [M, N] IoU matrix.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)), dtype=np.float64)

    g = np.asarray(gt_boxes, dtype=np.float64)
    p = np.asarray(pred_boxes, dtype=np.float64)

    x1 = np.maximum(g[:, 0:1], p[:, 0:1].T)
    y1 = np.maximum(g[:, 1:2], p[:, 1:2].T)
    x2 = np.minimum(g[:, 2:3], p[:, 2:3].T)
    y2 = np.minimum(g[:, 3:4], p[:, 3:4].T)

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])

    union = area_g[:, None] + area_p[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def compute_mota(
    gt_boxes_per_frame: List[np.ndarray],
    gt_ids_per_frame: List[np.ndarray],
    pred_boxes_per_frame: List[np.ndarray],
    pred_ids_per_frame: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Compute Multi-Object Tracking Accuracy (MOTA).

    MOTA = 1 - (FN + FP + IDSW) / total_gt

    Args:
        gt_boxes_per_frame: List of [M, 4] GT boxes per frame.
        gt_ids_per_frame: List of [M] GT identity IDs per frame.
        pred_boxes_per_frame: List of [N, 4] predicted boxes per frame.
        pred_ids_per_frame: List of [N] predicted track IDs per frame.
        iou_threshold: IoU threshold for matching.

    Returns:
        MOTA score.
    """
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0

    prev_matches: Dict[int, int] = {}

    for gt_boxes, gt_ids, pred_boxes, pred_ids in zip(
        gt_boxes_per_frame, gt_ids_per_frame, pred_boxes_per_frame, pred_ids_per_frame
    ):
        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        total_gt += n_gt

        if n_gt == 0:
            total_fp += n_pred
            continue
        if n_pred == 0:
            total_fn += n_gt
            continue

        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)

        matched_gt = set()
        matched_pred = set()
        current_matches: Dict[int, int] = {}

        for _ in range(min(n_gt, n_pred)):
            best_val = iou_mat.max()
            if best_val < iou_threshold:
                break
            gi, pi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)

            matched_gt.add(gi)
            matched_pred.add(pi)
            current_matches[int(gt_ids[gi])] = int(pred_ids[pi])

            iou_mat[gi, :] = 0
            iou_mat[:, pi] = 0

        total_fn += n_gt - len(matched_gt)
        total_fp += n_pred - len(matched_pred)

        for gt_id, pred_id in current_matches.items():
            if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                total_idsw += 1

        prev_matches = current_matches

    if total_gt == 0:
        return 0.0

    mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt
    return mota


def compute_motp(
    gt_boxes_per_frame: List[np.ndarray],
    pred_boxes_per_frame: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Compute Multi-Object Tracking Precision (MOTP).

    MOTP = average IoU of matched pairs.
    """
    total_iou = 0.0
    total_matches = 0

    for gt_boxes, pred_boxes in zip(gt_boxes_per_frame, pred_boxes_per_frame):
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)

        for _ in range(min(len(gt_boxes), len(pred_boxes))):
            best_val = iou_mat.max()
            if best_val < iou_threshold:
                break
            gi, pi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            total_iou += best_val
            total_matches += 1
            iou_mat[gi, :] = 0
            iou_mat[:, pi] = 0

    return total_iou / max(total_matches, 1)


def compute_idf1(
    gt_boxes_per_frame: List[np.ndarray],
    gt_ids_per_frame: List[np.ndarray],
    pred_boxes_per_frame: List[np.ndarray],
    pred_ids_per_frame: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """Compute IDF1 — the ratio of correctly identified detections.

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    """
    total_idtp = 0
    total_idfp = 0
    total_idfn = 0

    for gt_boxes, gt_ids, pred_boxes, pred_ids in zip(
        gt_boxes_per_frame, gt_ids_per_frame, pred_boxes_per_frame, pred_ids_per_frame
    ):
        if len(gt_boxes) == 0:
            total_idfp += len(pred_boxes)
            continue
        if len(pred_boxes) == 0:
            total_idfn += len(gt_boxes)
            continue

        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)

        matched_gt = set()
        matched_pred = set()

        for _ in range(min(len(gt_boxes), len(pred_boxes))):
            best_val = iou_mat.max()
            if best_val < iou_threshold:
                break
            gi, pi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            matched_gt.add(gi)
            matched_pred.add(pi)
            total_idtp += 1
            iou_mat[gi, :] = 0
            iou_mat[:, pi] = 0

        total_idfn += len(gt_boxes) - len(matched_gt)
        total_idfp += len(pred_boxes) - len(matched_pred)

    denom = 2 * total_idtp + total_idfp + total_idfn
    return (2 * total_idtp / denom) if denom > 0 else 0.0


def compute_id_switches(
    gt_ids_per_frame: List[np.ndarray],
    pred_ids_per_frame: List[np.ndarray],
    gt_boxes_per_frame: List[np.ndarray],
    pred_boxes_per_frame: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> int:
    """Count identity switches across frames."""
    prev_matches: Dict[int, int] = {}
    total_switches = 0

    for gt_boxes, gt_ids, pred_boxes, pred_ids in zip(
        gt_boxes_per_frame, gt_ids_per_frame, pred_boxes_per_frame, pred_ids_per_frame
    ):
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
        current_matches: Dict[int, int] = {}

        for _ in range(min(len(gt_boxes), len(pred_boxes))):
            best_val = iou_mat.max()
            if best_val < iou_threshold:
                break
            gi, pi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            current_matches[int(gt_ids[gi])] = int(pred_ids[pi])
            iou_mat[gi, :] = 0
            iou_mat[:, pi] = 0

        for gt_id, pred_id in current_matches.items():
            if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                total_switches += 1

        prev_matches = current_matches

    return total_switches


def compute_track_fragmentation(
    gt_ids_per_frame: List[np.ndarray],
    pred_ids_per_frame: List[np.ndarray],
    gt_boxes_per_frame: List[np.ndarray],
    pred_boxes_per_frame: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> int:
    """Count track fragmentations (a GT track matched then unmatched then matched again)."""
    gt_track_status: Dict[int, bool] = {}
    total_frags = 0

    for gt_boxes, gt_ids, pred_boxes, pred_ids in zip(
        gt_boxes_per_frame, gt_ids_per_frame, pred_boxes_per_frame, pred_ids_per_frame
    ):
        matched_gts = set()

        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
            for _ in range(min(len(gt_boxes), len(pred_boxes))):
                best_val = iou_mat.max()
                if best_val < iou_threshold:
                    break
                gi, pi = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                matched_gts.add(int(gt_ids[gi]))
                iou_mat[gi, :] = 0
                iou_mat[:, pi] = 0

        for gt_id in (int(g) for g in gt_ids):
            was_matched = gt_track_status.get(gt_id, False)
            is_matched = gt_id in matched_gts

            if was_matched and not is_matched:
                pass  # just lost
            elif not was_matched and is_matched and gt_id in gt_track_status:
                total_frags += 1

            gt_track_status[gt_id] = is_matched

    return total_frags
