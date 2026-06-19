"""Feature encoder that extracts and combines backbone features for ReID.

Takes multi-scale backbone features and produces a single feature map
suitable for the ReID head via lightweight convolution and pooling.
"""

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """Lightweight CNN encoder for extracting ReID-ready features.

    Takes the last stage output from ShuffleNetV2 and applies a small
    convolutional block with global average pooling to produce a compact
    feature vector.

    Args:
        in_channels: Number of input channels from backbone's last stage.
        out_channels: Output feature dimension before the ReID head.
    """

    BACKBONE_LAST_CHANNELS = {
        "0.5x": 192,
        "1.0x": 464,
        "1.5x": 704,
        "2.0x": 976,
    }

    def __init__(self, in_channels: int = 464, out_channels: int = 256):
        super().__init__()
        mid_channels = max(out_channels, in_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, features: list) -> torch.Tensor:
        """Encode backbone features into a flat vector.

        Args:
            features: List of backbone stage outputs. Uses the last one.

        Returns:
            Tensor of shape [B, out_channels].
        """
        x = features[-1]
        x = self.conv(x)
        x = self.gap(x)
        return x.flatten(1)
