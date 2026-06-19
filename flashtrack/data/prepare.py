"""Dataset preparation utilities for MOT format data."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def verify_dataset(root_dir: str) -> bool:
    """Verify that a MOT-format dataset exists and has expected structure.

    Checks for at least one sequence directory with ``img1/`` and ``gt/gt.txt``.

    Args:
        root_dir: Path to the MOT data root (e.g. ``data/MOT17/train``).

    Returns:
        True if the dataset appears valid.
    """
    root = Path(root_dir)
    if not root.exists():
        logger.warning("Dataset root does not exist: %s", root_dir)
        return False

    seq_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not seq_dirs:
        logger.warning("No sequence directories found in %s", root_dir)
        return False

    valid = 0
    for seq in seq_dirs:
        img_dir = seq / "img1"
        gt_file = seq / "gt" / "gt.txt"
        if img_dir.exists() and any(img_dir.glob("*.jpg")):
            valid += 1
            if gt_file.exists():
                logger.info("  ✓ %s (images + GT)", seq.name)
            else:
                logger.info("  ✓ %s (images only, no GT)", seq.name)

    if valid == 0:
        logger.warning("No valid sequences found (need img1/ with .jpg files)")
        return False

    logger.info("Dataset verified: %d/%d sequences valid", valid, len(seq_dirs))
    return True


def convert_mot_to_internal(
    src_dir: str,
    dst_dir: str,
    min_vis: float = 0.3,
    min_box_area: int = 100,
    split_ratio: float = 0.8,
) -> dict:
    """Convert a raw MOT-format dataset into train/val splits.

    Creates a clean copy with consistent naming and optionally filters
    annotations by visibility and box area.

    Args:
        src_dir: Source MOT directory (containing sequence sub-dirs).
        dst_dir: Destination directory.
        min_vis: Minimum visibility to keep a GT box.
        min_box_area: Minimum bbox area.
        split_ratio: Fraction of sequences for training.

    Returns:
        Dict with ``train_dir``, ``val_dir``, ``num_train``, ``num_val`` stats.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    train_dir = dst / "train"
    val_dir = dst / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    n_train = max(1, int(len(seq_dirs) * split_ratio))

    train_seqs = seq_dirs[:n_train]
    val_seqs = seq_dirs[n_train:]

    total_train_samples = 0
    total_val_samples = 0

    for seqs, out_dir, label in [(train_seqs, train_dir, "train"), (val_seqs, val_dir, "val")]:
        for seq in seqs:
            out_seq = out_dir / seq.name
            out_img = out_seq / "img1"
            out_gt_dir = out_seq / "gt"
            out_img.mkdir(parents=True, exist_ok=True)
            out_gt_dir.mkdir(parents=True, exist_ok=True)

            # Copy images
            src_img = seq / "img1"
            if src_img.exists():
                for img_file in sorted(src_img.glob("*.jpg")):
                    shutil.copy2(str(img_file), str(out_img / img_file.name))

            # Filter and copy GT
            src_gt = seq / "gt" / "gt.txt"
            if src_gt.exists():
                lines = src_gt.read_text().strip().split("\n")
                filtered = []
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) < 7:
                        continue
                    w, h = float(parts[4]), float(parts[5])
                    conf = float(parts[6])
                    vis = float(parts[8]) if len(parts) > 8 else 1.0

                    if conf > 0 and vis >= min_vis and w * h >= min_box_area:
                        filtered.append(line.strip())

                (out_gt_dir / "gt.txt").write_text("\n".join(filtered) + "\n")

                n_samples = len(filtered)
            else:
                n_samples = 0

            if label == "train":
                total_train_samples += n_samples
            else:
                total_val_samples += n_samples

            logger.info("  %s → %s (%d GT entries)", seq.name, label, n_samples)

    logger.info(
        "Conversion complete: %d train seqs (%d entries), %d val seqs (%d entries)",
        len(train_seqs), total_train_samples,
        len(val_seqs), total_val_samples,
    )

    return {
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "num_train_seqs": len(train_seqs),
        "num_val_seqs": len(val_seqs),
        "num_train_samples": total_train_samples,
        "num_val_samples": total_val_samples,
    }
