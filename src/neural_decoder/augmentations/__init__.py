"""Augmentation components."""
# Import from the parent module's augmentations.py file
# We use importlib to avoid circular import issues when we have both
# a file and directory with the same name
import importlib.util
from pathlib import Path

# Get the parent directory and import augmentations.py directly
_parent_dir = Path(__file__).parent.parent
_augmentations_file = _parent_dir / "augmentations.py"

if _augmentations_file.exists():
    spec = importlib.util.spec_from_file_location("_augmentations_module", _augmentations_file)
    if spec and spec.loader:
        _augmentations_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_augmentations_module)
        WhiteNoise = _augmentations_module.WhiteNoise
        MeanDriftNoise = _augmentations_module.MeanDriftNoise
        GaussianSmoothing = _augmentations_module.GaussianSmoothing
        TimeMaskingAugmentation = _augmentations_module.TimeMaskingAugmentation
        FeatureMaskingAugmentation = _augmentations_module.FeatureMaskingAugmentation
    else:
        raise ImportError("Could not load augmentations.py module")
else:
    raise ImportError(f"augmentations.py not found at {_augmentations_file}")

from .factory import create_augmentation_pipeline

__all__ = [
    "WhiteNoise",
    "MeanDriftNoise", 
    "GaussianSmoothing",
    "TimeMaskingAugmentation",
    "FeatureMaskingAugmentation",
    "create_augmentation_pipeline",
]

