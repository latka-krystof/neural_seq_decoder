"""Pipeline for chaining multiple preprocessors."""
import numpy as np
from typing import List
from .base import Preprocessor


class SequentialPreprocessor(Preprocessor):
    """Chain multiple preprocessors in sequence.
    
    Example:
        pipeline = SequentialPreprocessor([
            LogTransform(epsilon=1e-5),
            RobustScaler()
        ])
        transformed = pipeline.fit_transform(data)
    """
    
    def __init__(self, preprocessors: List[Preprocessor]):
        """Initialize sequential pipeline.
        
        Args:
            preprocessors: List of preprocessors to apply in order
        """
        self.preprocessors = preprocessors
    
    def fit(self, data: np.ndarray) -> None:
        """Fit all preprocessors in sequence.
        
        Args:
            data: Training data
        """
        current_data = data
        for preprocessor in self.preprocessors:
            preprocessor.fit(current_data)
            current_data = preprocessor.transform(current_data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply all preprocessors in sequence.
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        current_data = data
        for preprocessor in self.preprocessors:
            current_data = preprocessor.transform(current_data)
        return current_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform all preprocessors.
        
        Args:
            data: Training data
            
        Returns:
            Transformed data
        """
        current_data = data
        for preprocessor in self.preprocessors:
            current_data = preprocessor.fit_transform(current_data)
        return current_data

