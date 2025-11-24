"""Factory for creating optimizers."""
import torch
from typing import Dict, Any


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from configuration.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary with 'type' and 'params' keys
        
    Returns:
        Configured optimizer
        
    Example:
        config = {
            'type': 'adamw',
            'params': {
                'lr': 0.02,
                'weight_decay': 0.05,
                'eps': 1e-8
            }
        }
        optimizer = create_optimizer(model, config)
    """
    opt_type = config.get('type', 'adam').lower()
    params = config.get('params', {})
    
    if opt_type == 'adam':
        # Default Adam parameters matching baseline
        default_params = {
            'lr': 0.02,
            'betas': (0.9, 0.999),
            'eps': 0.1,  # Baseline uses large epsilon
            'weight_decay': 1e-5,
        }
        default_params.update(params)
        return torch.optim.Adam(model.parameters(), **default_params)
    
    elif opt_type == 'adamw':
        # AdamW with decoupled weight decay
        default_params = {
            'lr': 0.02,
            'betas': (0.9, 0.999),
            'eps': 1e-8,  # Standard PyTorch default (smaller than baseline)
            'weight_decay': 0.05,  # Higher weight decay recommended for AdamW
        }
        default_params.update(params)
        return torch.optim.AdamW(model.parameters(), **default_params)
    
    elif opt_type == 'sgd':
        default_params = {
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
        }
        default_params.update(params)
        return torch.optim.SGD(model.parameters(), **default_params)
    
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}. Choose from: adam, adamw, sgd")

