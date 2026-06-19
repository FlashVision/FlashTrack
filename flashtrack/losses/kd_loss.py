"""Knowledge Distillation losses for FlashTrack.

Adapted from FlashDet's KD losses for the ReID / tracking domain.
Supports embedding-level and logit-level distillation between a large
teacher ReID model and a smaller student.

Supported modes:
  - **Logit KD**: KL-divergence between teacher and student ID classification
    logits (soft targets).
  - **Embedding KD**: L2 alignment between teacher and student embeddings.
  - **Combined**: Both logit and embedding KD with configurable weighting.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LogitDistillationLoss(nn.Module):
    """Classification logit distillation via KL-divergence.

    Matches the teacher's soft class probability distribution using
    temperature-scaled KL divergence (Hinton et al., 2015).

    Args:
        temperature: Softmax temperature for KL divergence.
        weight: Overall loss weight.
    """

    def __init__(self, temperature: float = 4.0, weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> dict:
        """Compute logit-level KD loss.

        Args:
            student_logits: [B, C] student classification logits.
            teacher_logits: [B, C] teacher classification logits (detached).

        Returns:
            Dict with ``kd_logit_loss``.
        """
        T = self.temperature

        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        t_probs = F.softmax(teacher_logits / T, dim=-1)
        kd_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)

        return {
            "kd_logit_loss": self.weight * kd_loss,
        }


class EmbeddingDistillationLoss(nn.Module):
    """Embedding-level distillation via L2 / cosine alignment.

    Aligns student embeddings to teacher embeddings using normalised
    L2 distance or cosine similarity loss.

    Args:
        student_dim: Student embedding dimension.
        teacher_dim: Teacher embedding dimension.
        loss_weight: Overall weighting.
        distance: ``"l2"`` or ``"cosine"``.
    """

    def __init__(
        self,
        student_dim: int = 128,
        teacher_dim: int = 256,
        loss_weight: float = 0.5,
        distance: str = "l2",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.distance = distance

        if student_dim != teacher_dim:
            self.adapter = nn.Linear(student_dim, teacher_dim, bias=False)
        else:
            self.adapter = nn.Identity()

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute embedding-level KD loss.

        Args:
            student_embeddings: [B, D_s] student embeddings.
            teacher_embeddings: [B, D_t] teacher embeddings (detached).

        Returns:
            Scalar embedding distillation loss.
        """
        s_emb = self.adapter(student_embeddings)
        s_norm = F.normalize(s_emb, dim=1)
        t_norm = F.normalize(teacher_embeddings, dim=1)

        if self.distance == "cosine":
            loss = 1.0 - F.cosine_similarity(s_norm, t_norm, dim=1).mean()
        else:
            loss = F.mse_loss(s_norm, t_norm)

        return self.loss_weight * loss


class KnowledgeDistillationLoss(nn.Module):
    """Combined knowledge distillation loss for ReID models.

    Combines logit-level and embedding-level distillation.

    Args:
        temperature: KL divergence temperature.
        logit_weight: Weight for the logit KD component.
        embedding_weight: Weight for the embedding KD component.
        student_dim: Student embedding dimension.
        teacher_dim: Teacher embedding dimension.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        logit_weight: float = 1.0,
        embedding_weight: float = 0.5,
        student_dim: int = 128,
        teacher_dim: int = 256,
    ):
        super().__init__()
        self.logit_loss = LogitDistillationLoss(
            temperature=temperature,
            weight=logit_weight,
        )
        self.embedding_loss = EmbeddingDistillationLoss(
            student_dim=student_dim,
            teacher_dim=teacher_dim,
            loss_weight=embedding_weight,
        )
        self.logit_weight = logit_weight
        self.embedding_weight = embedding_weight

    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
        student_logits: torch.Tensor = None,
        teacher_logits: torch.Tensor = None,
    ) -> dict:
        """Compute the combined KD loss.

        Returns:
            Dict with all loss components and the combined ``kd_loss``.
        """
        result = {}

        if self.logit_weight > 0 and student_logits is not None and teacher_logits is not None:
            logit_res = self.logit_loss(student_logits, teacher_logits)
            result.update(logit_res)
        else:
            result["kd_logit_loss"] = torch.tensor(0.0, device=student_embeddings.device)

        if self.embedding_weight > 0:
            emb_loss = self.embedding_loss(student_embeddings, teacher_embeddings)
            result["kd_embedding_loss"] = emb_loss
        else:
            emb_loss = torch.tensor(0.0, device=student_embeddings.device)
            result["kd_embedding_loss"] = emb_loss

        result["kd_loss"] = result["kd_logit_loss"] + emb_loss

        return result
