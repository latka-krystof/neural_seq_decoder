"""Multi-head self-attention mechanism for neural sequence decoder.
Based on Transformer architecture, allowing the model to attend to different parts of the sequence.
"""
import torch
from torch import nn
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model, n_heads, dropout=0.0):
        """
        Args:
            d_model: Model dimension (input and output dimension)
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional mask tensor of shape (batch, seq_len) or (batch, seq_len, seq_len)
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Compute Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention: (batch, seq_len, n_heads, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, d_k)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, d_k)
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output


class AttentionBlock(nn.Module):
    """Transformer-style attention block with residual connections and layer normalization.
    
    Structure: LayerNorm -> MultiHeadAttention -> Residual -> LayerNorm -> FFN -> Residual
    """
    
    def __init__(self, d_model, n_heads, ff_dim=None, dropout=0.0):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            ff_dim: Feed-forward network dimension (default: 4 * d_model)
            dropout: Dropout probability
        """
        super(AttentionBlock, self).__init__()
        
        if ff_dim is None:
            ff_dim = 4 * d_model
        
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output = self.attention(x_norm, mask)
        x = x + attn_output  # Residual connection
        
        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output  # Residual connection
        
        return x

