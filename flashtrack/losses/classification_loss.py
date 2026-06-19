"""Cross-entropy loss with optional label smoothing for ID classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for person identity classification.

    Used alongside triplet loss during ReID training to provide identity
    supervision signal through the classification branch.

    Args:
        num_classes: Number of identity classes.
        label_smooth: Label smoothing factor (0 = no smoothing).
    """

    def __init__(self, num_classes: int = 500, label_smooth: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.label_smooth = label_smooth

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with optional label smoothing.

        Args:
            logits: [B, num_classes] raw classification logits.
            labels: [B] integer class labels.

        Returns:
            Scalar loss.
        """
        if self.label_smooth > 0:
            return self._label_smooth_ce(logits, labels)
        return F.cross_entropy(logits, labels)

    def _label_smooth_ce(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)

        targets = torch.zeros_like(log_probs)
        targets.fill_(self.label_smooth / (self.num_classes - 1))
        targets.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smooth)

        loss = (-targets * log_probs).sum(dim=1).mean()
        return loss
