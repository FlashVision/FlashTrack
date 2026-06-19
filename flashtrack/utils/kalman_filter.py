"""Kalman filter for bounding-box state estimation.

State vector: [cx, cy, ar, h, vx, vy, var, vh]
  - (cx, cy): centre of bounding box
  - ar: aspect ratio (width / height)
  - h: height
  - vx, vy, var, vh: respective velocities

Measurement vector: [cx, cy, ar, h]
"""

import numpy as np
import scipy.linalg


class KalmanFilter:
    """A simple linear Kalman filter for bounding-box tracking.

    Uses a constant-velocity model.  Designed for use in SORT, ByteTrack,
    and DeepSORT trackers.
    """

    _motion_mat = None
    _update_mat = None

    def __init__(self):
        ndim, dt = 4, 1.0

        if KalmanFilter._motion_mat is None:
            KalmanFilter._motion_mat = np.eye(2 * ndim, 2 * ndim)
            for i in range(ndim):
                KalmanFilter._motion_mat[i, ndim + i] = dt

            KalmanFilter._update_mat = np.eye(ndim, 2 * ndim)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        """Create track from an unassociated measurement.

        Args:
            measurement: [cx, cy, ar, h]

        Returns:
            (mean, covariance) of the new track.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman prediction step.

        Args:
            mean: [8] state mean.
            covariance: [8, 8] state covariance.

        Returns:
            (predicted_mean, predicted_covariance).
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Project state distribution to measurement space.

        Returns:
            (projected_mean, projected_covariance).
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        return projected_mean, projected_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman correction step.

        Args:
            mean: [8] predicted state mean.
            covariance: [8, 8] predicted state covariance.
            measurement: [4] observed (cx, cy, ar, h).

        Returns:
            (corrected_mean, corrected_covariance).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            chol,
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Compute gating distance (squared Mahalanobis) between state and measurements.

        Args:
            mean: [8] state mean.
            covariance: [8, 8] state covariance.
            measurements: [N, 4] measurements.
            only_position: Use only (cx, cy) for gating.

        Returns:
            [N] squared Mahalanobis distances.
        """
        proj_mean, proj_cov = self.project(mean, covariance)

        if only_position:
            proj_mean = proj_mean[:2]
            proj_cov = proj_cov[:2, :2]
            measurements = measurements[:, :2]

        chol = np.linalg.cholesky(proj_cov)
        d = measurements - proj_mean
        z = scipy.linalg.solve_triangular(
            chol, d.T, lower=True, check_finite=False,
        )
        return np.sum(z * z, axis=0)
