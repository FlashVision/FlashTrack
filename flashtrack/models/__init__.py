from .backbone import ShuffleNetV2
from .byte_tracker import ByteTracker
from .deepsort_tracker import DeepSORTTracker
from .encoder.feature_encoder import FeatureEncoder
from .head.reid_head import ReIDHead
from .sort_tracker import SORTTracker
from .tracker import FlashTracker, build_model

__all__ = [
    "FlashTracker", "build_model",
    "ShuffleNetV2", "FeatureEncoder", "ReIDHead",
    "ByteTracker", "SORTTracker", "DeepSORTTracker",
]

# Import new trackers (these live under flashtrack.trackers)
# to make them accessible from flashtrack.models namespace
try:
    from flashtrack.trackers import BoTSORTTracker, OCSORTTracker
    from flashtrack.trackers.sot import OSTrack, TemplateSearchTracker
    __all__ += ["BoTSORTTracker", "OCSORTTracker", "OSTrack", "TemplateSearchTracker"]
except ImportError:
    pass
