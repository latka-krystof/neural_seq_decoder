"""Focal CTC Loss for handling class imbalance in phoneme prediction."""
import torch
import torch.nn as nn
from typing import Optional


class FocalCTCLoss(nn.Module):
    """Focal loss variant for CTC to handle class imbalance.
    
    The focal loss down-weights easy examples (tokens the model predicts with
    high confidence) and focuses the gradient on hard examples (rare phonemes,
    difficult transitions). This is particularly useful for imbalanced datasets
    where silence tokens and vowels dominate.
    
    Formula: L_focal = (1 - p_t)^γ * L_ctc
    where p_t is the model's confidence (exp(-loss)) and γ is the focusing parameter.
    
    Args:
        blank: Index of the CTC blank token (default: 0)
        reduction: Reduction method ('mean', 'sum', or 'none')
        gamma: Focusing parameter. Higher gamma down-weights easy examples more (default: 2.0)
        alpha: Optional class-specific weighting (not used in current implementation)
        zero_infinity: Whether to zero out infinite losses (default: True)
    """
    
    def __init__(
        self,
        blank: int = 0,
        reduction: str = 'mean',
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        zero_infinity: bool = True
    ):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank,
            reduction='none',  # Get per-sample losses
            zero_infinity=zero_infinity
        )
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal CTC loss.
        
        Args:
            logits: Model predictions of shape (seq_len, batch, n_classes+1)
            targets: Target sequences of shape (batch, target_len)
            input_lengths: Length of each input sequence
            target_lengths: Length of each target sequence
            
        Returns:
            Scalar loss value (if reduction='mean' or 'sum') or per-sample losses
        """
        # Compute standard CTC loss per sample
        ctc_losses = self.ctc_loss(
            logits.log_softmax(2),
            targets,
            input_lengths,
            target_lengths
        )
        
        # Convert loss to confidence (probability)
        # Lower loss = higher confidence
        # We use exp(-loss) as a proxy for confidence
        p_t = torch.exp(-ctc_losses)
        p_t = torch.clamp(p_t, min=1e-8, max=1.0 - 1e-8)
        
        # Compute focal weighting: down-weight easy examples
        focal_weight = (1 - p_t) ** self.gamma
        
        # Optional: class-specific alpha weighting
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight
        
        # Apply focal weighting to CTC losses
        focal_loss = focal_weight * ctc_losses
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

