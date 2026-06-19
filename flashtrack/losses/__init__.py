from .classification_loss import ClassificationLoss
from .kd_loss import EmbeddingDistillationLoss, KnowledgeDistillationLoss
from .triplet_loss import TripletLoss

__all__ = [
    "TripletLoss",
    "ClassificationLoss",
    "KnowledgeDistillationLoss",
    "EmbeddingDistillationLoss",
]
