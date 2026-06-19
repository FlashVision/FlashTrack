"""Visualization utilities for multi-object tracking."""

import colorsys
from typing import Dict, Tuple

import cv2
import numpy as np

TRACK_COLORS: Dict[int, Tuple[int, int, int]] = {}


def _get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a unique, visually distinct BGR color for each track ID."""
    if track_id not in TRACK_COLORS:
        hue = (track_id * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.90)
        TRACK_COLORS[track_id] = (int(b * 255), int(g * 255), int(r * 255))
    return TRACK_COLORS[track_id]


def _font_params(img_h: int):
    """Return (font_scale, thickness, pad) scaled to image height."""
    base = max(img_h / 480.0, 0.4)
    scale = round(base * 0.45, 2)
    thickness = max(1, int(base * 0.9))
    pad = max(2, int(base * 3))
    return scale, thickness, pad


def draw_tracks(
    image: np.ndarray,
    tracks: list,
    trail_length: int = 30,
    trail_history: dict = None,
) -> np.ndarray:
    """Draw tracked objects with unique colors and optional trail lines.

    Args:
        image: BGR image.
        tracks: List of track dicts with ``track_id``, ``tlbr`` or ``tlwh``.
            Can also be STrack objects (from ByteTracker).
        trail_length: Number of past positions to draw as trail.
        trail_history: Mutable dict mapping track_id → list of centre points.
            Pass the same dict across frames to accumulate trails.

    Returns:
        Annotated BGR image.
    """
    output = image.copy()
    h = output.shape[0]
    font_scale, font_thick, pad = _font_params(h)
    lt = max(1, int(h / 300))

    if trail_history is None:
        trail_history = {}

    for t in tracks:
        if isinstance(t, dict):
            track_id = t["track_id"]
            if "tlbr" in t:
                x1, y1, x2, y2 = [int(v) for v in t["tlbr"]]
            elif "tlwh" in t:
                tw = t["tlwh"]
                x1, y1 = int(tw[0]), int(tw[1])
                x2, y2 = int(tw[0] + tw[2]), int(tw[1] + tw[3])
            else:
                continue
            score = t.get("score", 0.0)
        else:
            track_id = getattr(t, "track_id", 0)
            tlbr = getattr(t, "tlbr", None)
            if tlbr is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in tlbr]
            score = getattr(t, "score", 0.0)

        color = _get_track_color(track_id)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, lt)

        text = f"ID:{track_id}"
        if score > 0:
            text += f" {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        bg_tl = (x1, y1 - th - pad * 2)
        bg_br = (x1 + tw + pad * 2, y1)
        cv2.rectangle(output, bg_tl, bg_br, color, -1)
        cv2.putText(output, text, (x1 + pad, y1 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        # Trail
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if track_id not in trail_history:
            trail_history[track_id] = []
        trail_history[track_id].append((cx, cy))
        if len(trail_history[track_id]) > trail_length:
            trail_history[track_id].pop(0)

        points = trail_history[track_id]
        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(lt * alpha))
            cv2.line(output, points[i - 1], points[i], color, thickness, cv2.LINE_AA)

    return output


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray = None,
    scores: np.ndarray = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw simple bounding boxes on an image.

    Args:
        image: BGR image.
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format.
        labels: [N] text labels (optional).
        scores: [N] confidence scores (optional).
        color: BGR color.

    Returns:
        Annotated BGR image.
    """
    output = image.copy()
    h = output.shape[0]
    font_scale, font_thick, pad = _font_params(h)
    lt = max(1, int(h / 300))

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, lt)

        parts = []
        if labels is not None and i < len(labels):
            parts.append(str(labels[i]))
        if scores is not None and i < len(scores):
            parts.append(f"{scores[i]:.2f}")

        if parts:
            text = ": ".join(parts)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            cv2.rectangle(output, (x1, y1 - th - pad * 2), (x1 + tw + pad * 2, y1), color, -1)
            cv2.putText(output, text, (x1 + pad, y1 - pad),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return output


def add_fps_overlay(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Add FPS counter overlay to image."""
    cv2.putText(image, f"FPS: {fps:.1f}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return image
