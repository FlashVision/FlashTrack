"""Image transforms for ReID training, validation, and inference."""

import random
from typing import Tuple

import cv2
import numpy as np
import torch


class TrainTransform:
    """Training augmentation for ReID person crops.

    Applies random horizontal flip, color jitter, random erasing, and
    normalization.

    Args:
        crop_size: Target (H, W) for person crops.
        flip_prob: Horizontal flip probability.
        brightness: Max brightness shift.
        contrast: Contrast range (low, high).
        saturation: Saturation range (low, high).
        erasing_prob: Random erasing probability.
        mean: Normalization mean (RGB, 0-255 scale).
        std: Normalization std (RGB, 0-255 scale).
    """

    def __init__(
        self,
        crop_size: Tuple[int, int] = (128, 64),
        flip_prob: float = 0.5,
        brightness: float = 0.2,
        contrast: Tuple[float, float] = (0.8, 1.2),
        saturation: Tuple[float, float] = (0.8, 1.2),
        erasing_prob: float = 0.5,
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        std: Tuple[float, ...] = (58.395, 57.12, 57.375),
    ):
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.erasing_prob = erasing_prob
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """Transform an RGB image crop.

        Args:
            img: RGB uint8 image [H, W, 3].

        Returns:
            Normalised float32 tensor [3, crop_H, crop_W].
        """
        img = cv2.resize(img, (self.crop_size[1], self.crop_size[0]))

        if random.random() < self.flip_prob:
            img = np.fliplr(img).copy()

        img = img.astype(np.float32)

        # Brightness
        delta = random.uniform(-self.brightness, self.brightness) * 255
        img = np.clip(img + delta, 0, 255)

        # Contrast
        alpha = random.uniform(*self.contrast)
        img = np.clip(img * alpha, 0, 255)

        # Saturation (in HSV)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            s_factor = random.uniform(*self.saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # Normalize
        img = (img - self.mean) / self.std

        # Random erasing
        if random.random() < self.erasing_prob:
            img = self._random_erasing(img)

        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)

    @staticmethod
    def _random_erasing(img: np.ndarray, area_ratio=(0.02, 0.4), aspect=(0.3, 3.3)):
        h, w = img.shape[:2]
        area = h * w

        for _ in range(10):
            target_area = random.uniform(*area_ratio) * area
            aspect_ratio = random.uniform(*aspect)

            eh = int(round((target_area * aspect_ratio) ** 0.5))
            ew = int(round((target_area / aspect_ratio) ** 0.5))

            if eh < h and ew < w:
                y = random.randint(0, h - eh)
                x = random.randint(0, w - ew)
                img[y:y + eh, x:x + ew] = np.random.normal(0, 1, (eh, ew, 3)).astype(np.float32)
                break
        return img


class ValTransform:
    """Validation transform for ReID crops: resize + normalize."""

    def __init__(
        self,
        crop_size: Tuple[int, int] = (128, 64),
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        std: Tuple[float, ...] = (58.395, 57.12, 57.375),
    ):
        self.crop_size = crop_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (self.crop_size[1], self.crop_size[0]))
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)


class InferenceTransform:
    """Inference transform for full-frame or crop images.

    Resizes, normalises, and returns both the tensor and metadata for
    coordinate back-projection.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (128, 64),
        mean: Tuple[float, ...] = (123.675, 116.28, 103.53),
        std: Tuple[float, ...] = (58.395, 57.12, 57.375),
    ):
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray):
        """Process image for inference.

        Args:
            img: RGB uint8 [H, W, 3].

        Returns:
            (tensor [3, H', W'], meta dict with scale_h, scale_w).
        """
        h, w = img.shape[:2]
        th, tw = self.input_size

        resized = cv2.resize(img, (tw, th)).astype(np.float32)
        resized = (resized - self.mean) / self.std
        tensor = resized.transpose(2, 0, 1)

        meta = {
            "original_size": (h, w),
            "input_size": self.input_size,
            "scale_h": h / th,
            "scale_w": w / tw,
        }
        return tensor, meta
