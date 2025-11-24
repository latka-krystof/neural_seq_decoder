"""Base class for preprocessing components."""
from abc import ABC, abstractmethod
import numpy as np


class Preprocessor(ABC):
    """Abstract base class for data preprocessors."""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Learn statistics from training data.
        
        Args:
            data: Training data of shape [n_samples, n_features] or [n_samples, n_time, n_features]
        """
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply transformation to data.
        
        Args:
            data: Data to transform, same shape as fit data
            
        Returns:
            Transformed data with same shape
        """
        pass
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            data: Training data
            
        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data)

