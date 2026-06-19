"""MOT dataset for ReID training and tracking evaluation.

Reads MOT17-style data:
  <seq>/img1/<frame>.jpg   — images
  <seq>/gt/gt.txt          — ground truth (frame, id, x, y, w, h, conf, cls, vis)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MOTDataset(Dataset):
    """Dataset for MOT17-format tracking data.

    For ReID training, yields identity-labelled person crops.
    For tracking evaluation, yields full frames with ground-truth annotations.

    Args:
        root_dir: Path to MOT data root (e.g. ``data/MOT17/train``).
        mode: ``"reid"`` for ReID crop training, ``"tracking"`` for full-frame.
        transform: Callable transform applied to images/crops.
        min_vis: Minimum visibility threshold for ground-truth boxes.
        min_box_area: Minimum bbox area to include.
        crop_size: Crop size (H, W) for ReID mode.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "reid",
        transform=None,
        min_vis: float = 0.3,
        min_box_area: int = 100,
        crop_size: Tuple[int, int] = (128, 64),
    ):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transform = transform
        self.min_vis = min_vis
        self.min_box_area = min_box_area
        self.crop_size = crop_size

        if mode == "reid":
            self.samples = self._load_reid_samples()
            self._remap_ids()
        else:
            self.frames = self._load_tracking_frames()

    def _load_reid_samples(self) -> List[Dict]:
        """Load per-crop samples for ReID training."""
        samples = []
        for seq_dir in sorted(self.root_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_file = seq_dir / "gt" / "gt.txt"
            img_dir = seq_dir / "img1"
            if not gt_file.exists() or not img_dir.exists():
                continue

            for line in gt_file.read_text().strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = float(parts[6])

                vis = float(parts[8]) if len(parts) > 8 else 1.0

                if conf <= 0 or vis < self.min_vis or w * h < self.min_box_area:
                    continue

                img_path = img_dir / f"{frame_id:06d}.jpg"
                if not img_path.exists():
                    continue

                samples.append({
                    "img_path": str(img_path),
                    "bbox": [x, y, w, h],
                    "track_id": track_id,
                    "seq": seq_dir.name,
                })

        logger.info(f"Loaded {len(samples)} ReID samples from {self.root_dir}")
        return samples

    def _remap_ids(self):
        """Remap track IDs to contiguous 0..N-1."""
        unique_ids = sorted({s["track_id"] for s in self.samples})
        self.id_map = {old: new for new, old in enumerate(unique_ids)}
        self.num_ids = len(unique_ids)
        for s in self.samples:
            s["label"] = self.id_map[s["track_id"]]

    def _load_tracking_frames(self) -> List[Dict]:
        """Load per-frame data for tracking evaluation."""
        frames = []
        for seq_dir in sorted(self.root_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_file = seq_dir / "gt" / "gt.txt"
            img_dir = seq_dir / "img1"
            if not img_dir.exists():
                continue

            gt_data: Dict[int, List] = {}
            if gt_file.exists():
                for line in gt_file.read_text().strip().split("\n"):
                    parts = line.strip().split(",")
                    if len(parts) < 7:
                        continue
                    fid = int(parts[0])
                    tid = int(parts[1])
                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    if fid not in gt_data:
                        gt_data[fid] = []
                    gt_data[fid].append({"track_id": tid, "bbox": [x, y, w, h]})

            for img_path in sorted(img_dir.glob("*.jpg")):
                fid = int(img_path.stem)
                frames.append({
                    "img_path": str(img_path),
                    "frame_id": fid,
                    "seq": seq_dir.name,
                    "gt": gt_data.get(fid, []),
                })

        logger.info(f"Loaded {len(frames)} tracking frames from {self.root_dir}")
        return frames

    def __len__(self):
        return len(self.samples) if self.mode == "reid" else len(self.frames)

    def __getitem__(self, idx):
        if self.mode == "reid":
            return self._get_reid_item(idx)
        return self._get_tracking_item(idx)

    def _get_reid_item(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["img_path"])
        if img is None:
            img = np.zeros((self.crop_size[0], self.crop_size[1], 3), dtype=np.uint8)

        x, y, w, h = [int(v) for v in sample["bbox"]]
        ih, iw = img.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((self.crop_size[0], self.crop_size[1], 3), dtype=np.uint8)

        crop = cv2.resize(crop, (self.crop_size[1], self.crop_size[0]))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        if self.transform:
            crop = self.transform(crop)
        else:
            crop = crop.astype(np.float32).transpose(2, 0, 1) / 255.0
            crop = torch.from_numpy(crop)

        label = sample["label"]
        return crop, label

    def _get_tracking_item(self, idx):
        frame = self.frames[idx]
        img = cv2.imread(frame["img_path"])
        if img is None:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        gt_boxes = []
        gt_ids = []
        for ann in frame["gt"]:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_ids.append(ann["track_id"])

        gt_boxes = np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32)
        gt_ids = np.array(gt_ids, dtype=np.int64) if gt_ids else np.empty(0, dtype=np.int64)

        if self.transform:
            img = self.transform(img)

        return img, {
            "gt_boxes": gt_boxes,
            "gt_ids": gt_ids,
            "frame_id": frame["frame_id"],
            "seq": frame["seq"],
            "img_path": frame["img_path"],
        }
