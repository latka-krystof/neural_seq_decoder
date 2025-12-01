"""PyTorch transforms for neural data."""
import torch


class LogTransform:
    """Apply log(x + epsilon) transformation to PyTorch tensors.
    
    Neural firing rates and spike power typically follow a log-normal distribution.
    This transformation makes the data more amenable to neural network optimization.
    
    Note: Ideally, log transformation should be applied BEFORE z-score normalization
    during data preprocessing. This transform can be used at runtime for experimentation,
    but for best results, modify the notebook to apply log transform before normalization.
    """
    
    def __init__(self, epsilon: float = 1e-5):
        """Initialize log transform.
        
        Args:
            epsilon: Small constant to add before log to handle zeros and negative values
        """
        self.epsilon = epsilon
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply log transformation.
        
        Args:
            x: Input tensor (any shape)
            
        Returns:
            log(x + epsilon) with same shape
        """
        return torch.log(x + self.epsilon)

