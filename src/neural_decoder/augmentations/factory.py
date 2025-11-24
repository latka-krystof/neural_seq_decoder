"""Factory for creating augmentation pipelines."""
import torch.nn as nn
from typing import Dict, Any, Optional, List
# Import from the parent module's augmentations.py file directly
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


def create_augmentation_pipeline(config: Dict[str, Any]) -> Optional[nn.Module]:
    """Create augmentation pipeline from configuration.
    
    Args:
        config: Configuration dictionary with augmentation settings
        
    Returns:
        Sequential module with augmentations, or None if no augmentations enabled
        
    Example:
        config = {
            'white_noise': {'enabled': True, 'std': 1.0},
            'constant_offset': {'enabled': True, 'std': 0.2}
        }
        aug_pipeline = create_augmentation_pipeline(config)
    """
    augmentations: List[nn.Module] = []
    
    # White noise augmentation
    white_noise_cfg = config.get('white_noise', {})
    if white_noise_cfg.get('enabled', False):
        std = white_noise_cfg.get('std', 0.8)
        augmentations.append(WhiteNoise(std=std))
    
    # Constant offset (mean drift) augmentation
    constant_offset_cfg = config.get('constant_offset', {})
    if constant_offset_cfg.get('enabled', False):
        std = constant_offset_cfg.get('std', 0.2)
        augmentations.append(MeanDriftNoise(std=std))
    
    # Time masking augmentation (Phase 2)
    time_masking_cfg = config.get('time_masking', {})
    if time_masking_cfg.get('enabled', False):
        num_masks = time_masking_cfg.get('num_masks', 2)
        mask_width = time_masking_cfg.get('mask_width', 20)
        p = time_masking_cfg.get('p', 1.0)
        augmentations.append(TimeMaskingAugmentation(num_masks=num_masks, mask_width=mask_width, p=p))
    
    # Feature masking augmentation (Phase 2)
    feature_masking_cfg = config.get('feature_masking', {})
    if feature_masking_cfg.get('enabled', False):
        num_masks = feature_masking_cfg.get('num_masks', 2)
        mask_width = feature_masking_cfg.get('mask_width', 16)
        p = feature_masking_cfg.get('p', 1.0)
        augmentations.append(FeatureMaskingAugmentation(num_masks=num_masks, mask_width=mask_width, p=p))
    
    if len(augmentations) == 0:
        return None
    
    return nn.Sequential(*augmentations)

