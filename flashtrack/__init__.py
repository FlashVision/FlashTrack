"""FlashTrack — Ultra-lightweight real-time multi-object tracking."""

__version__ = "1.0.0"

from flashtrack.analytics import Benchmark, compute_hota
from flashtrack.cfg import get_config
from flashtrack.engine.exporter import Exporter
from flashtrack.engine.predictor import Predictor
from flashtrack.engine.trainer import Trainer
from flashtrack.engine.validator import Validator
from flashtrack.models.byte_tracker import ByteTracker
from flashtrack.models.deepsort_tracker import DeepSORTTracker
from flashtrack.models.lora import apply_lora, apply_qlora, merge_lora_weights
from flashtrack.models.sort_tracker import SORTTracker
from flashtrack.models.tracker import FlashTracker
from flashtrack.trackers import BoTSORTTracker, OCSORTTracker
from flashtrack.trackers.sot import OSTrack, TemplateSearchTracker
from flashtrack.utils.cmc import CameraMotionCompensator

__all__ = [
    "FlashTracker", "Trainer", "Validator", "Predictor", "Exporter",
    "apply_lora", "apply_qlora", "merge_lora_weights", "get_config",
    "ByteTracker", "SORTTracker", "DeepSORTTracker",
    "BoTSORTTracker", "OCSORTTracker",
    "OSTrack", "TemplateSearchTracker",
    "CameraMotionCompensator",
    "Benchmark", "compute_hota",
    "__version__",
]
