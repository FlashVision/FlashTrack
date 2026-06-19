"""Triplet loss with batch-hard mining for ReID training.

Computes the hardest positive and hardest negative within each batch
to form informative triplets, following Hermans et al. (2017)
"In Defense of the Triplet Loss for Person Re-Identification".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Batch-hard triplet loss.

    For each anchor, selects:
      - Hardest positive: same-identity sample with largest distance.
      - Hardest negative: different-identity sample with smallest distance.

    Args:
        margin: Triplet margin.
        distance: ``"euclidean"`` or ``"cosine"``.
        soft: Use soft-margin variant (log1p) instead of hinge.
    """

    def __init__(
        self,
        margin: float = 0.3,
        distance: str = "euclidean",
        soft: bool = False,
    ):
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.soft = soft

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute batch-hard triplet loss.

        Args:
            embeddings: [B, D] L2-normalised embeddings.
            labels: [B] identity labels.

        Returns:
            Scalar loss.
        """
        if self.distance == "cosine":
            dist_mat = 1.0 - F.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
            )
        else:
            dist_mat = torch.cdist(embeddings, embeddings, p=2)

        n = embeddings.size(0)
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_neg = ~mask_pos

        # Exclude self-comparisons
        mask_pos.fill_diagonal_(False)

        # Hardest positive: max distance among same-identity pairs
        dist_ap = dist_mat.clone()
        dist_ap[~mask_pos] = -1e9
        hardest_pos, _ = dist_ap.max(dim=1)

        # Hardest negative: min distance among different-identity pairs
        dist_an = dist_mat.clone()
        dist_an[~mask_neg] = 1e9
        hardest_neg, _ = dist_an.min(dim=1)

        # Filter anchors that have both positive and negative
        valid = (hardest_pos > -1e8) & (hardest_neg < 1e8)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        hardest_pos = hardest_pos[valid]
        hardest_neg = hardest_neg[valid]

        if self.soft:
            loss = torch.log1p(torch.exp(hardest_pos - hardest_neg))
        else:
            loss = F.relu(hardest_pos - hardest_neg + self.margin)

        return loss.mean()
