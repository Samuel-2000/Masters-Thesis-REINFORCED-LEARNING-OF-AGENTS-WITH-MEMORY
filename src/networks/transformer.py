"""
Transformer-based policy network
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .base import BaseNetwork, EmbeddingLayer, AttentionAggregator

from src.core.constants import VOCAB_SIZE, OBSERVATION_SIZE, ACTION_SIZE


class TransformerPolicyNet(BaseNetwork):
    """Transformer-based policy network"""


    
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 512,
                 observation_size: int = OBSERVATION_SIZE,
                 hidden_size: int = 512,
                 action_size: int = ACTION_SIZE,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 memory_size: int = 10,
                 use_auxiliary: bool = False):
        
        super().__init__(observation_size, action_size, hidden_size, use_auxiliary)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.memory_size = memory_size
        
        # Embedding
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        
        # Token aggregator
        self.aggregator = AttentionAggregator(embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Learnable memory tokens
        self.memory_tokens = nn.Parameter(
            torch.randn(1, memory_size, embed_dim) * 0.02
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_size)
        )
        
        # Auxiliary heads
        if use_auxiliary:
            self.energy_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_size // 4),
                nn.GELU(),
                nn.Linear(hidden_size // 4, 1)
            )
            
            self.observation_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, observation_size)
            )
    
    def forward(self,
                x: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                return_auxiliary: bool = False) -> torch.Tensor:
        """Forward pass"""
        B, T, K = x.shape
        
        # Embed and aggregate
        embedded = self.embedding(x)  # [B, T, K, D]
        aggregated = self.aggregator(embedded)  # [B, T, D]
        
        # Add memory tokens
        if memory is None:
            memory = self.memory_tokens.expand(B, -1, -1)
        
        transformer_input = torch.cat([memory, aggregated], dim=1)
        
        # Apply transformer
        transformer_output = self.transformer(transformer_input)
        
        # Split memory and observation outputs
        memory_out = transformer_output[:, :self.memory_size]
        obs_out = transformer_output[:, self.memory_size:]
        
        # Get policy logits
        logits = self.policy_head(obs_out)
        
        if return_auxiliary and self.use_auxiliary:
            energy_pred = self.energy_head(obs_out)
            obs_pred = self.observation_head(obs_out)
            return logits, energy_pred, obs_pred, memory_out
        
        return logits
    
    def reset_state(self, batch_size: int = 1):
        """Reset memory (not needed for stateless transformer)"""
        pass
    
    def get_config(self):
        """Get configuration for saving"""
        config = super().get_config()
        config.update({
            'vocab_size': self.embedding.embedding.num_embeddings,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'memory_size': self.memory_size,
            'dropout': self.transformer.layers[0].dropout.p,
        })
        return config