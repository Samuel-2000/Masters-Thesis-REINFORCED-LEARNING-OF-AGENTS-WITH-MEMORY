"""
Multi-memory network combining LSTM, Transformer, and cache - FIXED VERSION
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .base import BaseNetwork, EmbeddingLayer, AttentionAggregator
from .lstm import LSTMPolicyNet
from .transformer import TransformerPolicyNet

from src.core.constants import VOCAB_SIZE, OBSERVATION_SIZE, ACTION_SIZE


class NeuralCache(nn.Module):
    """Neural cache with content-based addressing and batch support"""
    
    def __init__(self, embed_dim: int, cache_size: int = 50):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cache_size = cache_size
        
        # Cache storage
        self.register_buffer('keys', torch.zeros(cache_size, embed_dim))
        self.register_buffer('values', torch.zeros(cache_size, embed_dim))
        self.register_buffer('usage', torch.zeros(cache_size))
        self.register_buffer('age', torch.zeros(cache_size))
        
        # Projections for addressing
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        
        # Usage decay
        self.decay = 0.95
        
    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve from cache"""
        B = query.shape[0]
        
        # Project query
        query_proj = self.query_proj(query)  # [B, D]
        keys_proj = self.key_proj(self.keys)  # [C, D]
        
        # Compute similarity
        scores = torch.matmul(query_proj, keys_proj.T) / (self.embed_dim ** 0.5)
        
        # Apply softmax with temperature
        attn = torch.softmax(scores, dim=-1)  # [B, C]
        
        # Retrieve values
        retrieved = torch.matmul(attn, self.values)  # [B, D]
        
        # Update usage and age
        self.usage = self.usage * self.decay + attn.mean(dim=0).detach()
        self.age += 1
        
        return retrieved, attn
    
    def write(self, key: torch.Tensor, value: torch.Tensor, priority: float = 1.0):
        """Write to cache - selective batch writing"""
        with torch.no_grad():
            # Handle batch writes
            if key.dim() == 2:  # [B, D] - batch of keys
                batch_size = key.shape[0]
                
                # Randomly select a few items from batch to write (avoid overfilling)
                write_count = min(3, batch_size)  # Write at most 3 items
                indices = torch.randperm(batch_size)[:write_count]
                
                for idx in indices:
                    # Use a combination of usage and age to select slot
                    # Prefer low usage and old age for replacement
                    scores = self.usage - 0.001 * self.age  # Lower is better
                    slot = torch.argmin(scores)
                    
                    self.keys[slot] = key[idx].detach()
                    self.values[slot] = value[idx].detach()
                    self.usage[slot] = 1.0 * priority
                    self.age[slot] = 0
            else:  # Single key [D]
                # Use a combination of usage and age
                scores = self.usage - 0.001 * self.age
                slot = torch.argmin(scores)
                
                self.keys[slot] = key.detach()
                self.values[slot] = value.detach()
                self.usage[slot] = 1.0 * priority
                self.age[slot] = 0
    
    def reset(self):
        """Reset cache"""
        self.keys.zero_()
        self.values.zero_()
        self.usage.zero_()
        self.age.zero_()


class MultiMemoryPolicyNet(BaseNetwork):
    """Network with multiple memory systems"""
    
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 512,
                 observation_size: int = OBSERVATION_SIZE,
                 hidden_size: int = 512,
                 action_size: int = ACTION_SIZE,
                 transformer_heads: int = 8,
                 transformer_layers: int = 3,
                 cache_size: int = 50,
                 use_auxiliary: bool = False):
        
        super().__init__(observation_size, action_size, hidden_size, use_auxiliary)
        
        self.embed_dim = embed_dim
        self.cache_size = cache_size
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.vocab_size = vocab_size
        
        # Shared embedding
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)
        
        # Multiple memory systems - modified to extract hidden states
        # LSTM memory (without final head)
        self.lstm_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm_pos_embed = nn.Parameter(torch.empty(observation_size, embed_dim))
        nn.init.normal_(self.lstm_pos_embed, mean=0.0, std=embed_dim ** -0.5)
        self.lstm_aggregator = nn.Sequential(
            nn.Linear(embed_dim * observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        
        # Transformer memory (without final head)
        self.transformer_embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
        # Transformer aggregator
        self.transformer_aggregator = AttentionAggregator(embed_dim)
        
        # Neural cache
        self.neural_cache = NeuralCache(embed_dim, cache_size)
        
        # Memory fusion - input size: hidden_size (lstm) + hidden_size (transformer) + embed_dim (cache)
        fusion_input_size = hidden_size + hidden_size + embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Auxiliary heads
        if use_auxiliary:
            self.energy_head = nn.Linear(hidden_size, 1)
            self.observation_head = nn.Linear(hidden_size, observation_size)
        
        # Write buffer for cache
        self.write_buffer: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        
        # Hidden states
        self.lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.current_batch_size: Optional[int] = None
        
    def forward(self,
                x: torch.Tensor,
                return_auxiliary: bool = False) -> torch.Tensor:
        """Forward pass through all memory systems"""
        B, T, K = x.shape

        # Ensure input is LongTensor for embedding layer
        if x.dtype != torch.long:
            x = x.long()
        
        # Get embeddings
        embedded = self.embedding(x)  # [B, T, K, D]
        
        # Aggregate tokens
        aggregated = embedded.mean(dim=2)  # [B, T, D]
        
        # LSTM memory
        lstm_out = self._forward_lstm(x)  # [B, T, hidden_size]
        
        # Transformer memory
        transformer_out = self._forward_transformer(x)  # [B, T, hidden_size]
        
        # Neural cache (use current observation)
        current_obs = aggregated[:, -1] if T > 1 else aggregated.squeeze(1)
        cache_out, cache_attn = self.neural_cache(current_obs)  # [B, D]
        cache_out = cache_out.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        
        # Cache writing decision - only during training
        if self.training and T > 1:
            # Only write if current observation isn't well represented
            max_attn = cache_attn.max(dim=-1)[0]
            for i in range(B):
                if max_attn[i] < 0.3:
                    # Calculate priority based on attention (lower attention = higher priority)
                    priority = 1.0 - max_attn[i].item()
                    self.write_buffer.append(
                        (current_obs[i].detach(), aggregated[i, -1].detach(), priority)
                    )
        
        # Fuse memories
        combined = torch.cat([lstm_out, transformer_out, cache_out], dim=-1)
        fused = self.fusion(combined)
        
        # Policy
        logits = self.policy_head(fused)
        
        # Return auxiliary predictions if requested
        if return_auxiliary and self.use_auxiliary:
            energy_pred = self.energy_head(fused)
            obs_pred = self.observation_head(fused)
            return logits, energy_pred, obs_pred
        
        return logits
    
    def _forward_lstm(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM to get hidden states"""
        B, T, K = x.shape
        
        # Embed tokens: [B, T, K, D]
        x_embed = self.lstm_embedding(x)
        
        # Add positional encoding
        x_embed = x_embed + self.lstm_pos_embed
        
        # Flatten and aggregate: [B, T, K*D] -> [B, T, H]
        x_flat = x_embed.view(B, T, -1)
        aggregated = self.lstm_aggregator(x_flat)
        
        # LSTM over the temporal dimension
        if self.lstm_hidden is None or self.current_batch_size != B:
            # Initialize hidden state with correct batch size
            h0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x.device)
            c0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=x.device)
            self.lstm_hidden = (h0, c0)
            self.current_batch_size = B
        
        lstm_out, self.lstm_hidden = self.lstm(aggregated, self.lstm_hidden)
        
        return lstm_out  # [B, T, hidden_size]
    
    def _forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer to get hidden states"""
        B, T, K = x.shape
        
        # Embed tokens
        x_embed = self.transformer_embedding(x)  # [B, T, K, D]
        
        # Aggregate tokens using attention
        aggregated = self.transformer_aggregator(x_embed)  # [B, T, D]
        
        # Apply transformer
        transformer_out = self.transformer(aggregated)  # [B, T, D]
        
        # Project to hidden_size
        projection = nn.Linear(self.embed_dim, self.hidden_size, device=x.device)
        hidden = projection(transformer_out)  # [B, T, hidden_size]
        
        return hidden
    
    def flush_cache_buffer(self):
        """Write buffered items to cache"""
        if not self.write_buffer:
            return
        
        # Group items by priority
        if len(self.write_buffer) > 0:
            # Take only top items by priority to avoid cache thrashing
            items_to_write = sorted(self.write_buffer, key=lambda x: x[2], reverse=True)
            items_to_write = items_to_write[:min(5, len(items_to_write))]
            
            # Stack keys and values for batch writing
            keys = torch.stack([item[0] for item in items_to_write])
            values = torch.stack([item[1] for item in items_to_write])
            priorities = torch.tensor([item[2] for item in items_to_write], device=keys.device)
            
            # Write batch to cache
            self.neural_cache.write(keys, values, priorities.mean().item())
        
        self.write_buffer.clear()
    
    def reset_state(self, batch_size: int = 1):
        """Reset all memory systems"""
        self.lstm_hidden = None
        self.current_batch_size = None
        self.neural_cache.reset()
        self.write_buffer.clear()
    
    def get_config(self):
        """Get configuration for saving"""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'cache_size': self.cache_size,
            'transformer_heads': self.transformer_heads,
            'transformer_layers': self.transformer_layers,
        })
        return config