"""ReID head: produces L2-normalized embeddings for re-identification.

During training, an optional classification head (FC → num_ids) enables
cross-entropy ID loss alongside triplet loss.  At inference the
classification branch is discarded and only the embedding is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDHead(nn.Module):
    """Embedding + optional ID classification head.

    Args:
        in_channels: Input feature dimension from the encoder.
        reid_dim: Output embedding dimension (128 or 256).
        num_ids: Number of identity classes for training. Set to 0 to
            disable the classification branch (inference mode).
    """

    def __init__(self, in_channels: int = 256, reid_dim: int = 128, num_ids: int = 0):
        super().__init__()
        self.reid_dim = reid_dim
        self.num_ids = num_ids

        self.bn = nn.BatchNorm1d(in_channels)
        self.fc = nn.Linear(in_channels, reid_dim, bias=False)
        self.bn_neck = nn.BatchNorm1d(reid_dim)
        self.bn_neck.bias.requires_grad_(False)

        if num_ids > 0:
            self.classifier = nn.Linear(reid_dim, num_ids, bias=False)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor):
        """Produce embeddings (and optionally logits).

        Args:
            x: [B, in_channels] feature vector from encoder.

        Returns:
            dict with:
              - "embedding": [B, reid_dim] L2-normalised embedding.
              - "bn_embedding": [B, reid_dim] BN-normalised embedding (for triplet).
              - "logits": [B, num_ids] classification logits (only if num_ids > 0).
        """
        x = self.bn(x)
        feat = self.fc(x)
        bn_feat = self.bn_neck(feat)

        embedding = F.normalize(feat, p=2, dim=1)

        result = {
            "embedding": embedding,
            "bn_embedding": bn_feat,
        }

        if self.classifier is not None:
            result["logits"] = self.classifier(bn_feat)

        return result
