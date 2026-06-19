"""FlashTrack Exporter — export ReID models to ONNX (and other formats)."""

import logging
import os
from typing import Optional, Tuple

import torch

from flashtrack.cfg import get_config
from flashtrack.models.tracker import FlashTracker

logger = logging.getLogger(__name__)


class Exporter:
    """Export a FlashTracker ReID model to ONNX format.

    Example::

        from flashtrack import Exporter

        exporter = Exporter(model_path="workspace/model_best_inference.pth")
        exporter.export_onnx("reid_model.onnx")
    """

    def __init__(
        self,
        model_path: str,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        self.model_path = model_path
        self._input_size_override = input_size

        cfg = get_config()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        backbone_size = cfg.model.backbone_size
        encoder_channels = cfg.model.encoder_channels
        reid_dim = cfg.model.reid_dim
        inp_size = cfg.model.input_size

        if "config" in checkpoint:
            ckpt_cfg = checkpoint["config"]
            backbone_size = ckpt_cfg.get("backbone_size", backbone_size)
            encoder_channels = ckpt_cfg.get("encoder_channels", encoder_channels)
            reid_dim = ckpt_cfg.get("reid_dim", reid_dim)
            inp_size = ckpt_cfg.get("input_size", inp_size)

        if input_size is not None:
            inp_size = input_size

        self.input_size = inp_size
        self.reid_dim = reid_dim

        self.model = FlashTracker(
            backbone_size=backbone_size,
            encoder_channels=encoder_channels,
            reid_dim=reid_dim,
            num_ids=0,
            pretrained=False,
        )

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {total_params:,} parameters")

    def export(
        self,
        output: str = "reid_model.onnx",
        simplify: bool = True,
        **kwargs,
    ) -> str:
        """Export model (convenience alias for export_onnx)."""
        return self.export_onnx(output_path=output, simplify=simplify, **kwargs)

    def export_onnx(
        self,
        output_path: str = "reid_model.onnx",
        opset_version: int = 11,
        simplify: bool = True,
        dynamic_batch: bool = True,
    ) -> str:
        """Export model to ONNX format.

        Args:
            output_path: Path for the output .onnx file.
            opset_version: ONNX opset version.
            simplify: Whether to run onnxsim simplification.
            dynamic_batch: Whether to use dynamic batch axis.

        Returns:
            Path to the exported ONNX file.
        """
        inp_h, inp_w = self.input_size if isinstance(self.input_size, tuple) else (self.input_size, self.input_size)
        dummy_input = torch.randn(1, 3, inp_h, inp_w)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch"},
                "embedding": {0: "batch"},
            }

        class _ExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model.extract(x)

        wrapper = _ExportWrapper(self.model)

        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes=dynamic_axes,
            keep_initializers_as_inputs=True,
        )
        logger.info(f"ONNX exported: {output_path}")

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                onnx_model = onnx.load(output_path)
                simplified, _ = onnx_simplify(onnx_model)
                onnx.save(simplified, output_path)
                logger.info("ONNX model simplified successfully")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Output: {output_path} ({file_size:.2f} MB)")
        return output_path
