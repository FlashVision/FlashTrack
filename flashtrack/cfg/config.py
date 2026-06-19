"""
Configuration for FlashTrack Model.

Default paths point to MOT17 format data.  The trainer reads annotation
files automatically, so this is only a fallback.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DataConfig:
    """Dataset paths — point to your MOT-format data directory.

    MOT format expects:
      <seq>/img1/<frame>.jpg   — images
      <seq>/gt/gt.txt          — ground truth annotations
    """
    train_images: str = "data/MOT17/train"
    train_annotations: str = "data/MOT17/train"
    val_images: str = "data/MOT17/val"
    val_annotations: str = "data/MOT17/val"
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Official FlashTrack model specifications:
    - FlashTrack-m:      backbone=1.0x, reid_dim=128, ~0.8M params
    - FlashTrack-m-1.5x: backbone=1.5x, reid_dim=256, ~1.5M params
    - FlashTrack-m-0.5x: backbone=0.5x, reid_dim=128, ~0.3M params
    """
    name: str = "FlashTrack"
    num_ids: int = 500
    input_size: Tuple[int, int] = (128, 64)

    backbone: str = "ShuffleNetV2"
    backbone_size: str = "1.0x"
    backbone_pretrained: bool = True

    reid_dim: int = 128
    encoder_channels: int = 256


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 0.0003
    weight_decay: float = 0.0005
    warmup_epochs: int = 5
    grad_clip: float = 10.0
    val_interval: int = 5
    save_dir: str = "workspace/tracking_experiment"
    resume: Optional[str] = None

    # torchtune-inspired optimizations
    enable_activation_checkpointing: bool = False
    enable_activation_offloading: bool = False
    optimizer_in_bwd: bool = False
    use_8bit_optimizer: bool = False
    compile_model: bool = False

    # LoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["backbone", "encoder"])

    # QLoRA
    use_qlora: bool = False
    qlora_quant_dtype: str = "int8"

    # Knowledge Distillation
    use_kd: bool = False
    kd_teacher_checkpoint: Optional[str] = None
    kd_teacher_model_size: str = "m-1.5x"
    kd_temperature: float = 4.0
    kd_feature_weight: float = 0.5

    # Loss weights
    triplet_weight: float = 1.0
    classification_weight: float = 0.5
    triplet_margin: float = 0.3


@dataclass
class AugmentConfig:
    """Data augmentation configuration."""
    scale: Tuple[float, float] = (0.9, 1.1)
    flip_prob: float = 0.5
    brightness: float = 0.2
    contrast: Tuple[float, float] = (0.8, 1.2)
    saturation: Tuple[float, float] = (0.8, 1.2)
    random_erasing_prob: float = 0.5
    normalize_mean: List[float] = field(default_factory=lambda: [123.675, 116.28, 103.53])
    normalize_std: List[float] = field(default_factory=lambda: [58.395, 57.12, 57.375])


@dataclass
class Config:
    """Top-level configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)


MODEL_SIZE_MAP = {
    "m-0.5x": ("0.5x", 128, 128),
    "m": ("1.0x", 128, 256),
    "m-1.5x": ("1.5x", 256, 384),
}


def get_config(
    model_size: str = "m",
    input_size: Tuple[int, int] = (128, 64),
    num_ids: int = 500,
    **overrides,
) -> Config:
    """Return configuration for a given model size.

    Args:
        model_size: One of "m-0.5x", "m", "m-1.5x".
        input_size: Input image dimension (height, width) for ReID crops.
        num_ids: Number of identity classes for training.
        **overrides: Additional overrides applied to the Config.
    """
    cfg = Config()

    if model_size in MODEL_SIZE_MAP:
        backbone_size, reid_dim, encoder_channels = MODEL_SIZE_MAP[model_size]
        cfg.model.backbone_size = backbone_size
        cfg.model.reid_dim = reid_dim
        cfg.model.encoder_channels = encoder_channels

    cfg.model.input_size = input_size
    cfg.model.num_ids = num_ids

    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return cfg


def load_yaml_config(yaml_path: str) -> Config:
    """Load configuration from a YAML file.

    YAML structure mirrors the Config dataclass hierarchy:
        model:
          backbone_size: "1.0x"
          num_ids: 750
          input_size: [128, 64]
        data:
          train_images: data/MOT17/train
        train:
          epochs: 100
    """
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    if "model" in raw:
        for key, value in raw["model"].items():
            if key == "input_size" and isinstance(value, list):
                value = tuple(value)
            if hasattr(cfg.model, key):
                setattr(cfg.model, key, value)

    if "data" in raw:
        for key, value in raw["data"].items():
            if hasattr(cfg.data, key):
                setattr(cfg.data, key, value)

    if "train" in raw:
        for key, value in raw["train"].items():
            if hasattr(cfg.train, key):
                setattr(cfg.train, key, value)

    if "augment" in raw:
        for key, value in raw["augment"].items():
            if key in ("scale", "contrast", "saturation") and isinstance(value, list):
                value = tuple(value)
            if hasattr(cfg.augment, key):
                setattr(cfg.augment, key, value)

    # Derive reid_dim / encoder_channels from backbone_size if not explicitly set
    if "model" in raw and "reid_dim" not in raw["model"]:
        bs = cfg.model.backbone_size
        size_key = {"0.5x": "m-0.5x", "1.0x": "m", "1.5x": "m-1.5x"}.get(bs)
        if size_key and size_key in MODEL_SIZE_MAP:
            _, reid_dim, enc_ch = MODEL_SIZE_MAP[size_key]
            cfg.model.reid_dim = reid_dim
            cfg.model.encoder_channels = enc_ch

    return cfg
