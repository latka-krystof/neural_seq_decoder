"""Log transformation for neural data."""
import numpy as np
from .base import Preprocessor


class LogTransform(Preprocessor):
    """Apply log(x + epsilon) transformation to gaussianize heavy-tailed distributions.
    
    Neural firing rates and spike power typically follow a log-normal distribution.
    This transformation makes the data more amenable to neural network optimization.
    """
    
    def __init__(self, epsilon: float = 1e-5):
        """Initialize log transform.
        
        Args:
            epsilon: Small constant to add before log to handle zeros
        """
        self.epsilon = epsilon
    
    def fit(self, data: np.ndarray) -> None:
        """No-op for log transform (no statistics to learn)."""
        pass
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply log transformation.
        
        Args:
            data: Input data (can be any shape)
            
        Returns:
            log(data + epsilon) with same shape
        """
        return np.log(data + self.epsilon)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform (just applies transform)."""
        return self.transform(data)

