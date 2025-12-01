"""Factory for creating learning rate schedulers."""
import torch
from typing import Dict, Any, Optional
from torch.optim import Optimizer


def create_scheduler(
    optimizer: Optimizer, 
    config: Dict[str, Any],
    total_steps: Optional[int] = None
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary with 'type' and 'params' keys
        total_steps: Total number of training steps (required for some schedulers)
        
    Returns:
        Configured scheduler or None if type is 'none'
        
    Example:
        config = {
            'type': 'cosine_warmup',
            'params': {
                'warmup_steps': 1000,
                'total_steps': 10000
            }
        }
        scheduler = create_scheduler(optimizer, config, total_steps=10000)
    """
    sched_type = config.get('type', 'linear').lower()
    params = config.get('params', {})
    
    if sched_type == 'none' or sched_type is None:
        return None
    
    elif sched_type == 'linear':
        # Linear decay (baseline)
        default_params = {
            'start_factor': 1.0,
            'end_factor': 0.1,  # Will be overridden by lrEnd/lrStart ratio
        }
        default_params.update(params)
        return torch.optim.lr_scheduler.LinearLR(optimizer, **default_params)
    
    elif sched_type == 'cosine':
        # Cosine annealing
        if total_steps is None:
            raise ValueError("total_steps required for cosine scheduler")
        default_params = {
            'T_max': total_steps,
            'eta_min': 0.0,
        }
        default_params.update(params)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **default_params)
    
    elif sched_type == 'cosine_warmup':
        # Cosine annealing with linear warmup
        if total_steps is None:
            raise ValueError("total_steps required for cosine_warmup scheduler")
        
        warmup_steps = params.get('warmup_steps', 1000)
        T_max = params.get('T_max', total_steps - warmup_steps)
        eta_min = params.get('eta_min', 0.0)
        
        # Create warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,  # Start at 1% of max LR
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create cosine scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        
        # Chain them together
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    
    elif sched_type == 'cosine_restarts':
        # Cosine annealing with warm restarts
        default_params = {
            'T_0': 1000,
            'T_mult': 1,
            'eta_min': 0.0,
        }
        default_params.update(params)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **default_params
        )
    
    elif sched_type == 'warmup_constant':
        # Linear warmup followed by constant learning rate
        if total_steps is None:
            raise ValueError("total_steps required for warmup_constant scheduler")
        
        warmup_steps = params.get('warmup_steps', 200)
        start_factor = params.get('start_factor', 0.02)  # Start at 2% of max LR
        
        # Create warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create constant scheduler (LambdaLR that returns 1.0)
        constant_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0
        )
        
        # Chain them together
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[warmup_steps]
        )
    
    else:
        raise ValueError(
            f"Unknown scheduler type: {sched_type}. "
            "Choose from: none, linear, cosine, cosine_warmup, cosine_restarts, warmup_constant"
        )

