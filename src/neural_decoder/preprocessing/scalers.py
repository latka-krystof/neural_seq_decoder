"""Scaling/normalization components for neural data."""
import numpy as np
from .base import Preprocessor


class StandardScaler(Preprocessor):
    """Standard Z-score normalization (mean=0, std=1).
    
    This is the baseline scaling method used in the original code.
    """
    
    def __init__(self):
        """Initialize standard scaler."""
        self.mean_ = None
        self.std_ = None
    
    def fit(self, data: np.ndarray) -> None:
        """Compute mean and standard deviation.
        
        Args:
            data: Training data of shape [n_samples, n_features] or [n_samples, n_time, n_features]
        """
        # Handle 2D (samples, features) or 3D (samples, time, features)
        if data.ndim == 2:
            self.mean_ = np.mean(data, axis=0, keepdims=True)
            self.std_ = np.std(data, axis=0, keepdims=True)
        elif data.ndim == 3:
            # Flatten time dimension for statistics
            n_samples, n_time, n_features = data.shape
            data_flat = data.reshape(-1, n_features)
            self.mean_ = np.mean(data_flat, axis=0, keepdims=True)
            self.std_ = np.std(data_flat, axis=0, keepdims=True)
        else:
            raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
        
        # Avoid division by zero
        self.std_ = np.maximum(self.std_, 1e-8)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply Z-score normalization.
        
        Args:
            data: Data to transform, same shape as fit data
            
        Returns:
            Normalized data: (data - mean) / std
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Must call fit() before transform()")
        
        return (data - self.mean_) / self.std_


class RobustScaler(Preprocessor):
    """Robust scaling using median and IQR (interquartile range).
    
    More robust to outliers than standard scaling. Uses median instead of mean
    and IQR instead of standard deviation.
    """
    
    def __init__(self):
        """Initialize robust scaler."""
        self.median_ = None
        self.iqr_ = None
    
    def fit(self, data: np.ndarray) -> None:
        """Compute median and IQR.
        
        Args:
            data: Training data of shape [n_samples, n_features] or [n_samples, n_time, n_features]
        """
        # Handle 2D (samples, features) or 3D (samples, time, features)
        if data.ndim == 2:
            self.median_ = np.median(data, axis=0, keepdims=True)
            q75 = np.percentile(data, 75, axis=0, keepdims=True)
            q25 = np.percentile(data, 25, axis=0, keepdims=True)
            self.iqr_ = q75 - q25
        elif data.ndim == 3:
            # Flatten time dimension for statistics
            n_samples, n_time, n_features = data.shape
            data_flat = data.reshape(-1, n_features)
            self.median_ = np.median(data_flat, axis=0, keepdims=True)
            q75 = np.percentile(data_flat, 75, axis=0, keepdims=True)
            q25 = np.percentile(data_flat, 25, axis=0, keepdims=True)
            self.iqr_ = q75 - q25
        else:
            raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
        
        # Avoid division by zero
        self.iqr_ = np.maximum(self.iqr_, 1e-8)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply robust scaling.
        
        Args:
            data: Data to transform, same shape as fit data
            
        Returns:
            Scaled data: (data - median) / IQR
        """
        if self.median_ is None or self.iqr_ is None:
            raise ValueError("Must call fit() before transform()")
        
        return (data - self.median_) / self.iqr_

