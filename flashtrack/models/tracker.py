"""FlashTracker — ReID feature extraction model.

Architecture:  ShuffleNetV2 backbone → FeatureEncoder → ReIDHead
The model produces L2-normalised embeddings for person bounding-box crops,
suitable for multi-object tracking via appearance matching.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn

from flashtrack.models.backbone.shufflenet import ShuffleNetV2
from flashtrack.models.encoder.feature_encoder import FeatureEncoder
from flashtrack.models.head.reid_head import ReIDHead

logger = logging.getLogger(__name__)

MODEL_SIZE_MAP = {
    "0.5x": {"backbone_size": "0.5x", "encoder_channels": 128, "reid_dim": 128},
    "1.0x": {"backbone_size": "1.0x", "encoder_channels": 256, "reid_dim": 128},
    "1.5x": {"backbone_size": "1.5x", "encoder_channels": 384, "reid_dim": 256},
}

BACKBONE_LAST_CHANNELS = {
    "0.5x": 192,
    "1.0x": 464,
    "1.5x": 704,
    "2.0x": 976,
}


class FlashTracker(nn.Module):
    """Lightweight ReID feature extractor for multi-object tracking.

    Forward pass takes person crops and returns embeddings.  During training
    the ReID head also outputs identity logits for cross-entropy loss.

    Args:
        backbone_size: ShuffleNetV2 width multiplier ("0.5x", "1.0x", "1.5x").
        encoder_channels: Output channels for the feature encoder.
        reid_dim: Embedding dimension.
        num_ids: Number of identity classes (0 = inference-only, no classifier).
        pretrained: Load ImageNet pretrained backbone weights.
        input_size: Expected input (H, W) for ReID crops.
    """

    def __init__(
        self,
        backbone_size: str = "1.0x",
        encoder_channels: int = 256,
        reid_dim: int = 128,
        num_ids: int = 0,
        pretrained: bool = True,
        input_size: Tuple[int, int] = (128, 64),
    ):
        super().__init__()
        self.backbone_size = backbone_size
        self.encoder_channels = encoder_channels
        self.reid_dim = reid_dim
        self.num_ids = num_ids
        self.input_size = input_size

        in_channels = BACKBONE_LAST_CHANNELS.get(backbone_size, 464)

        self.backbone = ShuffleNetV2(model_size=backbone_size, pretrained=pretrained)
        self.encoder = FeatureEncoder(in_channels=in_channels, out_channels=encoder_channels)
        self.head = ReIDHead(
            in_channels=encoder_channels,
            reid_dim=reid_dim,
            num_ids=num_ids,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract ReID features from input crops.

        Args:
            x: Input tensor [B, 3, H, W] — person crops.
            return_logits: If True and num_ids > 0, also return ID logits.

        Returns:
            Dict with "embeddings" [B, reid_dim] and optionally "logits" [B, num_ids].
        """
        features = self.backbone(x)
        encoded = self.encoder(features)
        result = self.head(encoded)

        output = {"embeddings": result["embedding"]}
        if return_logits and "logits" in result:
            output["logits"] = result["logits"]
        return output

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method returning only the embedding tensor."""
        return self.forward(x)["embeddings"]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ReID features for bounding-box crops (inference mode).

        Args:
            x: Batch of person crops [B, 3, H, W].

        Returns:
            L2-normalised embeddings [B, reid_dim].
        """
        self.eval()
        return self.extract(x)

    def get_model_info(self) -> Dict[str, object]:
        """Return a summary of model architecture and parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())

        size_mb = total_params * 4 / (1024 ** 2)

        info = {
            "name": "FlashTracker",
            "backbone_size": self.backbone_size,
            "encoder_channels": self.encoder_channels,
            "reid_dim": self.reid_dim,
            "num_ids": self.num_ids,
            "input_size": self.input_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "backbone_params": backbone_params,
            "encoder_params": encoder_params,
            "head_params": head_params,
            "model_size_mb": round(size_mb, 2),
        }

        logger.info(
            "FlashTracker-%s: %.2fM params (%.2f MB), "
            "backbone=%d, encoder=%d, head=%d, reid_dim=%d",
            self.backbone_size,
            total_params / 1e6,
            size_mb,
            backbone_params,
            encoder_params,
            head_params,
            self.reid_dim,
        )
        return info


def build_model(
    config=None,
    backbone_size: str = "1.0x",
    encoder_channels: int = 256,
    reid_dim: int = 128,
    num_ids: int = 0,
    pretrained: bool = True,
    input_size: Tuple[int, int] = (128, 64),
) -> FlashTracker:
    """Factory function for building a FlashTracker model.

    Can be called with a Config object or with keyword arguments directly.
    """
    if config is not None:
        backbone_size = getattr(config.model, "backbone_size", backbone_size)
        encoder_channels = getattr(config.model, "encoder_channels", encoder_channels)
        reid_dim = getattr(config.model, "reid_dim", reid_dim)
        num_ids = getattr(config.model, "num_ids", num_ids)
        pretrained = getattr(config.model, "pretrained", pretrained)
        input_size = getattr(config.model, "input_size", input_size)

    return FlashTracker(
        backbone_size=backbone_size,
        encoder_channels=encoder_channels,
        reid_dim=reid_dim,
        num_ids=num_ids,
        pretrained=pretrained,
        input_size=input_size,
    )
