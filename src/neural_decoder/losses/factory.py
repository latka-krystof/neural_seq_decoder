"""Factory for creating loss functions."""
import torch.nn as nn
from typing import Dict, Any
from .focal_ctc import FocalCTCLoss


def create_loss(config: Dict[str, Any]) -> nn.Module:
    """Create loss function from configuration.
    
    Args:
        config: Configuration dictionary with 'type' and 'params' keys
        
    Returns:
        Configured loss function
        
    Example:
        config = {
            'type': 'focal_ctc',
            'params': {
                'blank': 0,
                'gamma': 2.0,
                'reduction': 'mean'
            }
        }
        loss = create_loss(config)
    """
    loss_type = config.get('type', 'ctc').lower()
    params = config.get('params', {})
    
    if loss_type == 'ctc':
        # Standard CTC loss
        default_params = {
            'blank': 0,
            'reduction': 'mean',
            'zero_infinity': True,
        }
        default_params.update(params)
        return nn.CTCLoss(**default_params)
    
    elif loss_type == 'focal_ctc':
        # Focal CTC loss
        default_params = {
            'blank': 0,
            'reduction': 'mean',
            'gamma': 2.0,
            'alpha': None,
            'zero_infinity': True,
        }
        default_params.update(params)
        return FocalCTCLoss(**default_params)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from: ctc, focal_ctc")

