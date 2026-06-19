"""FlashTrack Predictor — run multi-object tracking on video."""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from flashtrack.cfg import get_config
from flashtrack.data.transforms import InferenceTransform
from flashtrack.models.byte_tracker import ByteTracker
from flashtrack.models.deepsort_tracker import DeepSORTTracker
from flashtrack.models.sort_tracker import SORTTracker
from flashtrack.models.tracker import FlashTracker
from flashtrack.utils.visualization import draw_tracks

logger = logging.getLogger(__name__)

TRACKER_MAP = {
    "byte": ByteTracker,
    "sort": SORTTracker,
    "deepsort": DeepSORTTracker,
}


class Predictor:
    """High-level tracking inference wrapper.

    Runs a detector's output through a multi-object tracker, optionally
    using a FlashTracker ReID model for appearance-based association.

    Example::

        from flashtrack import Predictor

        pred = Predictor(model_path="workspace/model_best_inference.pth")
        pred.process_video("test.mp4", output_dir="output")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        tracker_type: str = "byte",
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        input_size: Tuple[int, int] = (128, 64),
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tracker_type = tracker_type
        self.input_size = input_size

        self.reid_model = None
        if model_path is not None:
            self.reid_model = self._load_reid_model(model_path)

        if tracker_type == "byte":
            self.tracker = ByteTracker(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
            )
        elif tracker_type == "sort":
            self.tracker = SORTTracker(
                max_age=track_buffer,
                iou_threshold=1.0 - match_thresh,
            )
        elif tracker_type == "deepsort":
            self.tracker = DeepSORTTracker(max_age=track_buffer * 2)
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")

        self.transform = InferenceTransform(input_size=input_size)

    def _load_reid_model(self, model_path: str) -> FlashTracker:
        cfg = get_config()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        backbone_size = cfg.model.backbone_size
        encoder_channels = cfg.model.encoder_channels
        reid_dim = cfg.model.reid_dim

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
            encoder_channels = ckpt_cfg.get("encoder_channels", encoder_channels)
            reid_dim = ckpt_cfg.get("reid_dim", reid_dim)

        model = FlashTracker(
            backbone_size=backbone_size,
            encoder_channels=encoder_channels,
            reid_dim=reid_dim,
            num_ids=0,
            pretrained=False,
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(self.device).eval()
        logger.info(f"ReID model loaded from {model_path}")
        return model

    @torch.no_grad()
    def extract_features(self, frame: np.ndarray, boxes_tlwh: np.ndarray) -> np.ndarray:
        """Extract ReID features for bounding-box crops.

        Args:
            frame: BGR image.
            boxes_tlwh: [N, 4] bounding boxes (top-left-width-height).

        Returns:
            [N, reid_dim] feature matrix.
        """
        if self.reid_model is None or len(boxes_tlwh) == 0:
            return np.empty((len(boxes_tlwh), 128), dtype=np.float32)

        crops = []
        h, w = frame.shape[:2]
        for box in boxes_tlwh:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[0] + box[2]))
            y2 = min(h, int(box[1] + box[3]))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                crop = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)

            crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            crop = crop.transpose(2, 0, 1) / 255.0
            crops.append(crop)

        batch = torch.from_numpy(np.stack(crops)).to(self.device)
        features = self.reid_model.predict(batch)
        return features.cpu().numpy()

    def process_video(
        self,
        video_path: str,
        output_dir: str = "output",
        show: bool = False,
        det_callback=None,
    ) -> str:
        """Process video file with tracking.

        Args:
            video_path: Path to input video.
            output_dir: Directory for output video.
            show: Display live preview.
            det_callback: Optional callable(frame) → (boxes_tlwh, scores, class_ids).
                If None, requires external detections.

        Returns:
            Path to the output tracked video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, Path(video_path).stem + "_tracked.mp4")
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
        )

        self.tracker.reset()
        frame_count = 0
        total_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()

            if det_callback is not None:
                boxes, scores, class_ids = det_callback(frame)
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                scores = np.empty(0, dtype=np.float32)
                class_ids = np.empty(0, dtype=int)

            if self.tracker_type == "deepsort" and self.reid_model is not None and len(boxes) > 0:
                features = self.extract_features(frame, boxes)
                tracks = self.tracker.update(boxes, scores, features=features, class_ids=class_ids)
            else:
                if hasattr(self.tracker, 'update'):
                    tracks = self.tracker.update(boxes, scores, class_ids=class_ids)
                else:
                    tracks = []

            total_time += time.time() - start

            output = draw_tracks(frame, tracks)
            current_fps = frame_count / total_time if total_time > 0 else 0
            cv2.putText(output, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            writer.write(output)
            frame_count += 1

            if show:
                cv2.imshow("FlashTrack", output)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 100 == 0:
                logger.info(f"  {frame_count}/{total} frames processed")

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = frame_count / total_time if total_time > 0 else 0
        logger.info(f"Video processed: {avg_fps:.1f} FPS, saved to {output_path}")
        return output_path
