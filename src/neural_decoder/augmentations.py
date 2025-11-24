import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    """Add a constant offset (mean drift) to each sample.
    
    For 3D tensors (batch, time, features), adds the same offset to all time steps
    for each sample. This simulates baseline drift in neural recordings.
    """
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        """Apply mean drift noise.
        
        Args:
            x: Input tensor of shape (batch, time, features) or (batch, features)
            
        Returns:
            x + noise, where noise is constant across time for each sample
        """
        if x.dim() == 3:
            # 3D: (batch, time, features) - add constant offset per sample
            B, T, F = x.shape
            noise = torch.randn(B, 1, F, device=x.device, dtype=x.dtype) * self.std
            return x + noise
        elif x.dim() == 2:
            # 2D: (batch, features) - add constant offset per sample
            B, F = x.shape
            noise = torch.randn(B, 1, device=x.device, dtype=x.dtype) * self.std
            return x + noise
        else:
            raise ValueError(f"MeanDriftNoise expects 2D or 3D input, got {x.dim()}D")

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")


class TimeMaskingAugmentation(nn.Module):
    """Mask contiguous time steps (SpecAugment-style for neural signals).
    
    Randomly masks contiguous blocks of time steps, forcing the model to rely
    on temporal context. This improves robustness to temporal variations and
    strengthens the internal language model.
    
    Args:
        num_masks: Number of time masks to apply
        mask_width: Width of each mask in time steps
        p: Probability of applying augmentation (default: 1.0 during training)
    """
    
    def __init__(self, num_masks=2, mask_width=20, p=1.0):
        super().__init__()
        self.num_masks = num_masks
        self.mask_width = mask_width
        self.p = p
    
    def forward(self, x):
        """
        Apply time masking.
        
        Args:
            x: Input tensor of shape (batch, time, features)
            
        Returns:
            Masked tensor with same shape
        """
        if not self.training or torch.rand(1, device=x.device) > self.p:
            return x
        
        batch_size, time_steps, n_features = x.shape
        
        # Create a mask tensor (MPS-compatible: fully vectorized, no in-place ops on views)
        mask = torch.ones(batch_size, time_steps, n_features, device=x.device, dtype=x.dtype)
        
        # Apply multiple masks using vectorized approach
        for _ in range(self.num_masks):
            # Random start position for each sample in batch
            max_start = max(1, time_steps - self.mask_width)
            t0 = torch.randint(0, max_start, (batch_size,), device=x.device)
            
            # Create boolean mask using vectorized operations (MPS-compatible)
            # Create time indices for all samples: [batch, time]
            t_indices = torch.arange(time_steps, device=x.device).unsqueeze(0).expand(batch_size, -1)
            # For each sample, mask the region [t0, t0+mask_width)
            mask_regions = (t_indices >= t0.unsqueeze(1)) & (t_indices < (t0 + self.mask_width).unsqueeze(1))
            # Apply mask (set to 0 where mask_regions is True) - expand to match features
            mask = mask * (~mask_regions.unsqueeze(2)).float()
        
        return x * mask


class FeatureMaskingAugmentation(nn.Module):
    """Mask contiguous features/channels (SpecAugment-style for neural signals).
    
    Randomly masks contiguous blocks of features, simulating electrode dropout
    and forcing the model to learn distributed representations that don't
    over-rely on any single channel.
    
    Args:
        num_masks: Number of feature masks to apply
        mask_width: Width of each mask in features
        p: Probability of applying augmentation (default: 1.0 during training)
    """
    
    def __init__(self, num_masks=2, mask_width=16, p=1.0):
        super().__init__()
        self.num_masks = num_masks
        self.mask_width = mask_width
        self.p = p
    
    def forward(self, x):
        """
        Apply feature masking.
        
        Args:
            x: Input tensor of shape (batch, time, features)
            
        Returns:
            Masked tensor with same shape
        """
        if not self.training or torch.rand(1, device=x.device) > self.p:
            return x
        
        batch_size, time_steps, n_features = x.shape
        
        # Create a mask tensor (MPS-compatible: fully vectorized, no in-place ops on views)
        mask = torch.ones(batch_size, time_steps, n_features, device=x.device, dtype=x.dtype)
        
        # Apply multiple masks using vectorized approach
        for _ in range(self.num_masks):
            # Random start position for each sample in batch
            max_start = max(1, n_features - self.mask_width)
            f0 = torch.randint(0, max_start, (batch_size,), device=x.device)
            
            # Create boolean mask using vectorized operations (MPS-compatible)
            # Create feature indices for all samples: [batch, features]
            f_indices = torch.arange(n_features, device=x.device).unsqueeze(0).expand(batch_size, -1)
            # For each sample, mask the region [f0, f0+mask_width)
            mask_regions = (f_indices >= f0.unsqueeze(1)) & (f_indices < (f0 + self.mask_width).unsqueeze(1))
            # Apply mask (set to 0 where mask_regions is True) - expand to match time
            mask = mask * (~mask_regions.unsqueeze(1)).float()
        
        return x * mask
