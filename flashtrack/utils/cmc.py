"""Camera Motion Compensation (CMC) for multi-object tracking.

Provides methods to estimate and compensate for camera ego-motion:
  - Affine transformation via sparse optical flow + RANSAC.
  - Homography estimation via feature matching.
  - Dense optical flow-based pixel-level compensation.
"""

import logging
from enum import Enum
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CMCMethod(Enum):
    AFFINE = "affine"
    HOMOGRAPHY = "homography"
    OPTICAL_FLOW = "optical_flow"
    NONE = "none"


class CameraMotionCompensator:
    """Estimates and compensates for camera motion between consecutive frames.

    Args:
        method: CMC method — ``"affine"``, ``"homography"``, ``"optical_flow"``, or ``"none"``.
        max_features: Maximum features for sparse flow / feature matching.
        ransac_threshold: RANSAC reprojection threshold in pixels.
        downscale: Downscale factor for faster processing.
    """

    def __init__(
        self,
        method: str = "affine",
        max_features: int = 200,
        ransac_threshold: float = 5.0,
        downscale: float = 1.0,
    ):
        self.method = CMCMethod(method) if method != "none" else CMCMethod.NONE
        self.max_features = max_features
        self.ransac_threshold = ransac_threshold
        self.downscale = downscale

        self._prev_frame: Optional[np.ndarray] = None
        self._prev_keypoints = None

    def reset(self):
        """Reset internal state (call between videos)."""
        self._prev_frame = None
        self._prev_keypoints = None

    def compute(self, frame_gray: np.ndarray) -> np.ndarray:
        """Compute the warp matrix to compensate for camera motion.

        Args:
            frame_gray: Current frame as grayscale uint8 array (H, W).

        Returns:
            Warp matrix: 2x3 for affine, 3x3 for homography,
            identity if first frame or method is "none".
        """
        if self.method == CMCMethod.NONE or self._prev_frame is None:
            self._prev_frame = frame_gray.copy()
            if self.method == CMCMethod.HOMOGRAPHY:
                return np.eye(3, dtype=np.float64)
            return np.eye(2, 3, dtype=np.float64)

        if self.downscale != 1.0:
            h, w = frame_gray.shape[:2]
            new_h, new_w = int(h / self.downscale), int(w / self.downscale)
            prev_small = _resize(self._prev_frame, (new_w, new_h))
            curr_small = _resize(frame_gray, (new_w, new_h))
        else:
            prev_small = self._prev_frame
            curr_small = frame_gray

        if self.method == CMCMethod.AFFINE:
            warp = self._estimate_affine(prev_small, curr_small)
        elif self.method == CMCMethod.HOMOGRAPHY:
            warp = self._estimate_homography(prev_small, curr_small)
        elif self.method == CMCMethod.OPTICAL_FLOW:
            warp = self._estimate_optical_flow(prev_small, curr_small)
        else:
            warp = np.eye(2, 3, dtype=np.float64)

        if self.downscale != 1.0 and warp is not None:
            warp = self._scale_warp(warp, self.downscale)

        self._prev_frame = frame_gray.copy()
        return warp if warp is not None else np.eye(2, 3, dtype=np.float64)

    def apply_to_boxes(
        self,
        boxes_tlwh: np.ndarray,
        warp_matrix: np.ndarray,
    ) -> np.ndarray:
        """Apply the warp transformation to bounding boxes.

        Args:
            boxes_tlwh: [N, 4] boxes in (x, y, w, h) format.
            warp_matrix: 2x3 affine or 3x3 homography matrix.

        Returns:
            [N, 4] transformed boxes in tlwh format.
        """
        if len(boxes_tlwh) == 0:
            return boxes_tlwh

        boxes = np.asarray(boxes_tlwh, dtype=np.float64)
        tl = boxes[:, :2]
        br = tl + boxes[:, 2:]

        corners = np.stack([tl, br], axis=1)  # (N, 2, 2)
        N = corners.shape[0]

        transformed = []
        for i in range(N):
            pts = corners[i]  # (2, 2) — tl, br
            new_pts = self._warp_points(pts, warp_matrix)
            new_tl = new_pts.min(axis=0)
            new_br = new_pts.max(axis=0)
            transformed.append(np.concatenate([new_tl, new_br - new_tl]))

        return np.array(transformed)

    def apply_to_points(
        self,
        points: np.ndarray,
        warp_matrix: np.ndarray,
    ) -> np.ndarray:
        """Apply warp to a set of (x, y) points.

        Args:
            points: [N, 2] points.
            warp_matrix: 2x3 or 3x3 warp matrix.

        Returns:
            [N, 2] transformed points.
        """
        return self._warp_points(points, warp_matrix)

    @staticmethod
    def _warp_points(
        points: np.ndarray,
        warp: np.ndarray,
    ) -> np.ndarray:
        """Apply affine/homography transform to points."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        N = pts.shape[0]
        ones = np.ones((N, 1))
        pts_h = np.concatenate([pts, ones], axis=1)  # (N, 3)

        if warp.shape[0] == 3:
            # Homography
            result_h = pts_h @ warp.T
            result = result_h[:, :2] / np.maximum(result_h[:, 2:3], 1e-8)
        else:
            # Affine (2x3)
            result = pts_h @ warp.T

        return result

    def _estimate_affine(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Estimate 2x3 affine warp via sparse optical flow + RANSAC."""
        try:
            import cv2
        except ImportError:
            return self._estimate_affine_numpy(prev_gray, curr_gray)

        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=self.max_features,
            qualityLevel=0.01, minDistance=30, blockSize=3,
        )
        if prev_pts is None or len(prev_pts) < 4:
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None,
        )
        if curr_pts is None:
            return None

        mask = status.flatten() == 1
        p0 = prev_pts[mask].reshape(-1, 2)
        p1 = curr_pts[mask].reshape(-1, 2)

        if len(p0) < 4:
            return None

        warp, _ = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )
        return warp

    def _estimate_affine_numpy(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Pure-numpy fallback for affine estimation using block matching."""
        h, w = prev_gray.shape[:2]
        block_size = 32
        search_range = 16

        src_pts = []
        dst_pts = []

        for by in range(0, h - block_size, block_size * 2):
            for bx in range(0, w - block_size, block_size * 2):
                template = prev_gray[by:by + block_size, bx:bx + block_size].astype(np.float64)

                sy1 = max(0, by - search_range)
                sx1 = max(0, bx - search_range)
                sy2 = min(h, by + block_size + search_range)
                sx2 = min(w, bx + block_size + search_range)

                search_area = curr_gray[sy1:sy2, sx1:sx2].astype(np.float64)

                sh, sw = search_area.shape
                if sh < block_size or sw < block_size:
                    continue

                best_sad = float("inf")
                best_dy, best_dx = 0, 0

                for dy in range(0, sh - block_size + 1, 4):
                    for dx in range(0, sw - block_size + 1, 4):
                        candidate = search_area[dy:dy + block_size, dx:dx + block_size]
                        sad = np.abs(template - candidate).sum()
                        if sad < best_sad:
                            best_sad = sad
                            best_dy, best_dx = dy, dx

                match_y = sy1 + best_dy
                match_x = sx1 + best_dx
                src_pts.append([bx + block_size / 2, by + block_size / 2])
                dst_pts.append([match_x + block_size / 2, match_y + block_size / 2])

        if len(src_pts) < 3:
            return None

        src = np.array(src_pts)
        dst = np.array(dst_pts)

        # Least-squares affine estimation
        N = len(src)
        A = np.zeros((2 * N, 6))
        b = np.zeros(2 * N)
        for i in range(N):
            A[2 * i] = [src[i, 0], src[i, 1], 1, 0, 0, 0]
            A[2 * i + 1] = [0, 0, 0, src[i, 0], src[i, 1], 1]
            b[2 * i] = dst[i, 0]
            b[2 * i + 1] = dst[i, 1]

        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        warp = np.array([
            [result[0], result[1], result[2]],
            [result[3], result[4], result[5]],
        ])
        return warp

    def _estimate_homography(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Estimate 3x3 homography via feature matching."""
        try:
            import cv2
        except ImportError:
            return np.eye(3, dtype=np.float64)

        orb = cv2.ORB_create(nfeatures=self.max_features)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return np.eye(3, dtype=np.float64)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) < 4:
            return np.eye(3, dtype=np.float64)

        src_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float64)
        dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float64)

        H, mask = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )
        return H if H is not None else np.eye(3, dtype=np.float64)

    def _estimate_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Estimate affine from dense optical flow."""
        try:
            import cv2
        except ImportError:
            return None

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        h, w = flow.shape[:2]
        step = max(h, w) // 20

        src_pts = []
        dst_pts = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                dx, dy = flow[y, x]
                if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                    continue
                src_pts.append([x, y])
                dst_pts.append([x + dx, y + dy])

        if len(src_pts) < 4:
            return None

        src = np.array(src_pts, dtype=np.float64)
        dst = np.array(dst_pts, dtype=np.float64)

        warp, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )
        return warp

    @staticmethod
    def _scale_warp(warp: np.ndarray, scale: float) -> np.ndarray:
        """Adjust warp matrix for downscaling."""
        if warp.shape == (2, 3):
            warp = warp.copy()
            warp[0, 2] *= scale
            warp[1, 2] *= scale
        elif warp.shape == (3, 3):
            S = np.diag([scale, scale, 1.0])
            S_inv = np.diag([1.0 / scale, 1.0 / scale, 1.0])
            warp = S @ warp @ S_inv
        return warp


def _resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Simple nearest-neighbor resize without cv2."""
    target_w, target_h = size
    h, w = image.shape[:2]
    row_idx = (np.arange(target_h) * h / target_h).astype(int)
    col_idx = (np.arange(target_w) * w / target_w).astype(int)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    return image[np.ix_(row_idx, col_idx)]


def compose_warp(warp1: np.ndarray, warp2: np.ndarray) -> np.ndarray:
    """Compose two warp matrices (apply warp1 then warp2)."""
    if warp1.shape == (2, 3):
        w1 = np.vstack([warp1, [0, 0, 1]])
    else:
        w1 = warp1

    if warp2.shape == (2, 3):
        w2 = np.vstack([warp2, [0, 0, 1]])
    else:
        w2 = warp2

    result = w2 @ w1

    if warp1.shape == (2, 3) and warp2.shape == (2, 3):
        return result[:2]
    return result


def invert_warp(warp: np.ndarray) -> np.ndarray:
    """Compute the inverse warp matrix."""
    if warp.shape == (2, 3):
        w = np.vstack([warp, [0, 0, 1]])
        inv = np.linalg.inv(w)
        return inv[:2]
    return np.linalg.inv(warp)
