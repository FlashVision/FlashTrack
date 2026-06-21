"""OSTrack — One-Stream Single Object Tracker.

Template-search architecture using a shared ViT backbone with early
candidate elimination for efficient single-object tracking.

References:
    Ye et al., "Joint Feature Learning and Relation Modeling for
    Tracking: A One-Stream Framework", ECCV 2022.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Patch-based image tokenization."""

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class TransformerBlock(nn.Module):
    """Standard ViT block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class CandidateElimination(nn.Module):
    """Early candidate elimination module.

    Removes low-attention search tokens at intermediate layers
    to reduce computation for the remaining layers.

    Args:
        keep_ratio: Fraction of search tokens to keep.
    """

    def __init__(self, keep_ratio: float = 0.7):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(
        self,
        tokens: torch.Tensor,
        template_len: int,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: (B, T+S, D) concatenated template+search tokens.
            template_len: Number of template tokens.
            attn_weights: (B, T+S) attention weights for scoring.

        Returns:
            (filtered_tokens, keep_indices).
        """
        B, N, D = tokens.shape
        template = tokens[:, :template_len]
        search = tokens[:, template_len:]
        S = search.shape[1]
        keep_n = max(int(S * self.keep_ratio), 1)

        if attn_weights is not None:
            search_scores = attn_weights[:, template_len:]
        else:
            search_scores = search.norm(dim=-1)

        _, top_indices = search_scores.topk(keep_n, dim=1)
        top_indices_sorted = top_indices.sort(dim=1).values

        kept_search = torch.gather(
            search, 1,
            top_indices_sorted.unsqueeze(-1).expand(-1, -1, D),
        )

        filtered = torch.cat([template, kept_search], dim=1)
        return filtered, top_indices_sorted


class CorrelationHead(nn.Module):
    """Correlation-based prediction head for single-object tracking.

    Computes dense correlation between template and search features,
    then predicts center offset and bounding box size.

    Args:
        embed_dim: Feature dimension.
        search_size: Expected spatial size of search features.
    """

    def __init__(self, embed_dim: int = 256, search_size: int = 16):
        super().__init__()
        self.search_size = search_size

        self.center_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 1, 1),
        )

        self.size_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 2, 1),
        )

        self.offset_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 2, 1),
        )

    def forward(self, search_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            search_features: (B, S, D) search region features.

        Returns:
            Dict with ``'center'`` (B, 1, H, W), ``'size'`` (B, 2, H, W),
            ``'offset'`` (B, 2, H, W).
        """
        B, S, D = search_features.shape
        h = w = int(math.sqrt(S))
        if h * w < S:
            h = w = h + 1
        feat_1d = search_features.transpose(1, 2)  # (B, D, S)
        feat_2d = torch.nn.functional.adaptive_avg_pool1d(feat_1d, self.search_size ** 2)
        feat_2d = feat_2d.reshape(B, D, self.search_size, self.search_size)

        center = self.center_head(feat_2d).sigmoid()
        size = self.size_head(feat_2d).sigmoid()
        offset = self.offset_head(feat_2d)

        return {"center": center, "size": size, "offset": offset}


class OSTrack(nn.Module):
    """OSTrack — One-Stream Tracker.

    Shared ViT backbone processes template and search in a single stream.
    Includes optional candidate elimination for efficiency.

    Args:
        template_size: Template crop size.
        search_size: Search region crop size.
        patch_size: ViT patch size.
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        eliminate_layer: Layer at which to apply candidate elimination (-1 to disable).
        keep_ratio: Fraction of search tokens to keep after elimination.
    """

    def __init__(
        self,
        template_size: int = 128,
        search_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        eliminate_layer: int = 4,
        keep_ratio: float = 0.7,
    ):
        super().__init__()
        self.template_size = template_size
        self.search_size = search_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.template_patches = (template_size // patch_size) ** 2
        self.search_patches = (search_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(patch_size, 3, embed_dim)
        self.template_pos = nn.Parameter(torch.zeros(1, self.template_patches, embed_dim))
        self.search_pos = nn.Parameter(torch.zeros(1, self.search_patches, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.eliminate_layer = eliminate_layer
        self.candidate_elimination = CandidateElimination(keep_ratio) if eliminate_layer >= 0 else None

        self.head = CorrelationHead(
            embed_dim, search_size=search_size // patch_size,
        )

        nn.init.trunc_normal_(self.template_pos, std=0.02)
        nn.init.trunc_normal_(self.search_pos, std=0.02)

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: joint template-search feature extraction + prediction.

        Args:
            template: (B, 3, template_size, template_size) template crop.
            search: (B, 3, search_size, search_size) search region crop.

        Returns:
            Dict with ``'center'``, ``'size'``, ``'offset'`` heatmaps.
        """
        t_tokens = self.patch_embed(template) + self.template_pos
        s_tokens = self.patch_embed(search) + self.search_pos

        tokens = torch.cat([t_tokens, s_tokens], dim=1)

        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)

            if (
                self.candidate_elimination is not None
                and i == self.eliminate_layer
            ):
                tokens, _ = self.candidate_elimination(tokens, self.template_patches)

        search_out = tokens[:, self.template_patches:]
        return self.head(search_out)

    @torch.no_grad()
    def predict_bbox(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        search_bbox: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """Predict target bounding box in image coordinates.

        Args:
            template: (1, 3, T, T) template crop.
            search: (1, 3, S, S) search region crop.
            search_bbox: (x, y, w, h) of the search region in the image.

        Returns:
            (x, y, w, h) predicted target bounding box in image coordinates.
        """
        self.eval()
        outputs = self.forward(template, search)

        center_map = outputs["center"].squeeze(0).squeeze(0)
        size_map = outputs["size"].squeeze(0)
        offset_map = outputs["offset"].squeeze(0)

        fh, fw = center_map.shape
        max_idx = center_map.argmax()
        cy_idx = max_idx // fw
        cx_idx = max_idx % fw

        cx = (cx_idx.float() + offset_map[0, cy_idx, cx_idx]) / fw
        cy = (cy_idx.float() + offset_map[1, cy_idx, cx_idx]) / fh
        bw = size_map[0, cy_idx, cx_idx].item()
        bh = size_map[1, cy_idx, cx_idx].item()

        sx, sy, sw, sh = search_bbox
        pred_x = sx + cx.item() * sw - bw * sw / 2
        pred_y = sy + cy.item() * sh - bh * sh / 2
        pred_w = bw * sw
        pred_h = bh * sh

        return (pred_x, pred_y, pred_w, pred_h)


class TemplateSearchTracker:
    """High-level single-object tracker built on OSTrack.

    Manages template initialization, search region cropping,
    and iterative tracking across frames.

    Args:
        model: An ``OSTrack`` model instance.
        device: Torch device.
        search_factor: Search region size relative to target.
        template_factor: Template crop size relative to target.
        search_size: Model search input size.
        template_size: Model template input size.
    """

    def __init__(
        self,
        model: OSTrack,
        device: str = "cpu",
        search_factor: float = 4.0,
        template_factor: float = 2.0,
        search_size: int = 256,
        template_size: int = 128,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.search_factor = search_factor
        self.template_factor = template_factor
        self.search_size = search_size
        self.template_size = template_size

        self._template: Optional[torch.Tensor] = None
        self._target_bbox: Optional[Tuple[float, float, float, float]] = None

    def init(
        self,
        image: torch.Tensor,
        bbox: Tuple[float, float, float, float],
    ):
        """Initialize the tracker with the first frame.

        Args:
            image: (1, 3, H, W) first frame.
            bbox: (x, y, w, h) initial target bounding box.
        """
        self._target_bbox = bbox
        self._template = self._crop_and_resize(image, bbox, self.template_factor, self.template_size)

    def track(
        self,
        image: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """Track the target in a new frame.

        Args:
            image: (1, 3, H, W) new frame.

        Returns:
            (x, y, w, h) predicted target bounding box.
        """
        search_crop, search_bbox = self._get_search_region(image)
        pred_bbox = self.model.predict_bbox(self._template, search_crop, search_bbox)
        self._target_bbox = pred_bbox
        return pred_bbox

    def _crop_and_resize(
        self,
        image: torch.Tensor,
        bbox: Tuple[float, float, float, float],
        factor: float,
        target_size: int,
    ) -> torch.Tensor:
        """Crop a region around bbox and resize."""
        _, _, H, W = image.shape
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        s = max(w, h) * factor

        x1 = max(0, int(cx - s / 2))
        y1 = max(0, int(cy - s / 2))
        x2 = min(W, int(cx + s / 2))
        y2 = min(H, int(cy + s / 2))

        crop = image[:, :, y1:y2, x1:x2]
        crop = F.interpolate(crop, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return crop.to(self.device)

    def _get_search_region(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[float, float, float, float]]:
        """Crop search region around current target."""
        _, _, H, W = image.shape
        x, y, w, h = self._target_bbox
        cx, cy = x + w / 2, y + h / 2
        s = max(w, h) * self.search_factor

        x1 = max(0, int(cx - s / 2))
        y1 = max(0, int(cy - s / 2))
        x2 = min(W, int(cx + s / 2))
        y2 = min(H, int(cy + s / 2))

        crop = image[:, :, y1:y2, x1:x2]
        crop = F.interpolate(
            crop, size=(self.search_size, self.search_size),
            mode="bilinear", align_corners=False,
        )
        return crop.to(self.device), (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
