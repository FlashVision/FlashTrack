"""DataLoader factory for FlashTrack."""

import logging
from typing import Optional, Tuple

from torch.utils.data import DataLoader

from flashtrack.data.dataset import MOTDataset
from flashtrack.data.transforms import TrainTransform, ValTransform

logger = logging.getLogger(__name__)


def create_dataloader(
    root_dir: str,
    mode: str = "reid",
    batch_size: int = 64,
    crop_size: Tuple[int, int] = (128, 64),
    num_workers: int = 4,
    is_train: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    sampler=None,
) -> DataLoader:
    """Create a DataLoader for MOT data.

    Args:
        root_dir: Path to MOT data root.
        mode: ``"reid"`` for ReID training or ``"tracking"`` for evaluation.
        batch_size: Batch size.
        crop_size: (H, W) for ReID crops.
        num_workers: Number of data-loading workers.
        is_train: Use training transforms and shuffling.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop the last incomplete batch.
        sampler: Optional sampler (overrides shuffle).

    Returns:
        A configured DataLoader.
    """
    transform = TrainTransform(crop_size=crop_size) if is_train else ValTransform(crop_size=crop_size)

    dataset = MOTDataset(
        root_dir=root_dir,
        mode=mode,
        transform=transform,
        crop_size=crop_size,
    )

    shuffle = is_train and sampler is None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last and is_train,
        sampler=sampler,
    )

    logger.info(
        "DataLoader: %d samples, batch=%d, workers=%d, mode=%s, train=%s",
        len(dataset), batch_size, num_workers, mode, is_train,
    )
    return loader
