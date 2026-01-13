# src/networks/lstm.py - FIXED VERSION
"""
LSTM-based policy network (Simplified to match original)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .base import BaseNetwork

from src.core.constants import VOCAB_SIZE, OBSERVATION_SIZE, ACTION_SIZE


class LSTMPolicyNet(BaseNetwork):  # Inherit from BaseNetwork
    """LSTM-based policy networke"""
    

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,  # Was 20, now 19
                 embed_dim: int = 512,
                 observation_size: int = OBSERVATION_SIZE,  # Always 10
                 hidden_size: int = 512,
                 action_size: int = ACTION_SIZE,  # Always 6
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 use_auxiliary: bool = False):
        
        # Call parent constructor
        super().__init__(observation_size, action_size, hidden_size, use_auxiliary)
        
        # Store configuration - matching original
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=None,
        )
        
        # Learnable positional encodings for the K tokens inside an observation
        self.pos_embed = nn.Parameter(torch.empty(observation_size, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=embed_dim ** -0.5)
        
        # ConcatMLP-style aggregator (like original)
        self.aggregator = nn.Sequential(
            nn.Linear(embed_dim * observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # LSTM memory
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Policy head (logits) - OUTPUTS 6 ACTIONS
        self.head = nn.Linear(hidden_size, action_size)
        
        # Auxiliary heads (if needed)
        if use_auxiliary:
            self.energy_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            self.observation_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, observation_size)
            )
        
        # Hidden state - store as None initially
        self.hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.current_batch_size: Optional[int] = None
        
    def reset_state(self, batch_size: int = 1):
        """Reset LSTM hidden state"""
        self.hidden_state = None
        self.current_batch_size = None
    
    def forward(self, x: torch.Tensor, return_auxiliary: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        x : LongTensor [batch, seq, K]
        
        Returns
        -------
        logits : Tensor [batch, seq, action_size]
        """
        B, T, K = x.shape
        
        # Ensure input is LongTensor
        if x.dtype != torch.long:
            x = x.long()
        
        # Validate token range
        x_min, x_max = x.min().item(), x.max().item()
        if x_min < 0 or x_max >= self.vocab_size:
            raise ValueError(f"Input tokens out of range [0, {self.vocab_size-1}]: "
                           f"min={x_min}, max={x_max}")
        
        # Embed tokens: [B, T, K, D]
        x_embed = self.embedding(x)
        
        # Add positional encoding
        x_embed = x_embed + self.pos_embed  # broadcast (K, D) -> (B, T, K, D)
        
        # Flatten and aggregate: [B, T, K*D] -> [B, T, H]
        x_flat = x_embed.view(B, T, -1)
        aggregated = self.aggregator(x_flat)
        
        # LSTM over the temporal dimension
        if self.hidden_state is None or self.current_batch_size != B:
            # Initialize hidden state with correct batch size
            h0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x.device)
            c0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x.device)
            self.hidden_state = (h0, c0)
            self.current_batch_size = B
        
        out, self.hidden_state = self.lstm(aggregated, self.hidden_state)
        
        # Get logits - should be [B, T, 6] (6 actions)
        logits = self.head(out)
        
        # Return auxiliary predictions if requested
        if return_auxiliary and self.use_auxiliary:
            energy_pred = self.energy_head(out)
            obs_pred = self.observation_head(out)
            return logits, energy_pred, obs_pred
        
        return logits
    
    def get_config(self):
        """Get configuration for saving"""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        })
        return config