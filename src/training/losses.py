"""
Loss functions for RL training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any  # Added missing imports


class PolicyLoss:
    """Policy gradient loss with entropy regularization"""
    
    def __init__(self, 
                 gamma: float = 0.97,
                 entropy_coef: float = 0.01,
                 normalize_advantages: bool = True):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.normalize_advantages = normalize_advantages
    
    def __call__(self,
                 logits: torch.Tensor,
                 actions: torch.Tensor,
                 rewards: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy gradient loss
        
        Args:
            logits: [B, T, A] action logits
            actions: [B, T] action indices
            rewards: [B, T] rewards
            mask: [B, T] mask for valid steps (1=valid, 0=invalid)
        
        Returns:
            loss: policy loss + entropy regularization
            entropy: entropy value (for monitoring)
        """
        B, T, A = logits.shape
        
        # Compute action probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, A]
        
        # Get log probability of taken actions
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # [B, T]
        
        # Compute returns
        returns = self._compute_returns(rewards, mask)
        
        # Compute advantages
        advantages = self._compute_advantages(returns, mask)
        
        # Apply mask if provided
        if mask is not None:
            action_log_probs = action_log_probs * mask
            advantages = advantages * mask
            valid_count = mask.sum()
        else:
            valid_count = B * T
        
        # Policy loss
        policy_loss = -(action_log_probs * advantages.detach()).sum() / valid_count
        
        # Entropy regularization
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1)  # [B, T]
        
        if mask is not None:
            entropy = entropy * mask
        
        entropy_loss = -self.entropy_coef * entropy.sum() / valid_count
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        return total_loss, entropy.sum() / valid_count
    
    def _compute_returns(self, 
                        rewards: torch.Tensor, 
                        mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute discounted returns"""
        B, T = rewards.shape
        
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(B, device=rewards.device)
        
        # Compute returns from the end
        for t in reversed(range(T)):
            running_return = rewards[:, t] + self.gamma * running_return
            returns[:, t] = running_return
            
            # Apply mask if provided
            if mask is not None:
                running_return = running_return * mask[:, t]
        
        return returns
    
    def _compute_advantages(self,
                           returns: torch.Tensor,
                           mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute advantages (normalized by default)"""
        advantages = returns.clone()
        
        # Normalize advantages
        if self.normalize_advantages:
            if mask is not None:
                # Only consider masked values for normalization
                masked_returns = returns * mask
                valid_count = mask.sum()
                if valid_count > 0:
                    mean = masked_returns.sum() / valid_count
                    std = torch.sqrt((masked_returns - mean).pow(2).sum() / valid_count + 1e-8)
                    advantages = (advantages - mean) / (std + 1e-8)
            else:
                mean = returns.mean()
                std = returns.std()
                advantages = (advantages - mean) / (std + 1e-8)
        
        return advantages


class AuxiliaryLoss:
    """Auxiliary losses for self-supervised learning"""
    
    def __init__(self,
                 energy_coef: float = 0.1,
                 obs_coef: float = 0.05,
                 obs_prediction_type: str = 'classification'):
        """
        Args:
            energy_coef: weight for energy prediction loss
            obs_coef: weight for observation prediction loss
            obs_prediction_type: 'classification' or 'regression'
        """
        self.energy_coef = energy_coef
        self.obs_coef = obs_coef
        self.obs_prediction_type = obs_prediction_type
        
        if obs_prediction_type == 'classification':
            self.obs_criterion = nn.CrossEntropyLoss(reduction='mean')
        else:  # regression
            self.obs_criterion = nn.MSELoss(reduction='mean')
    
    def __call__(self,
                 energy_pred: torch.Tensor,
                 energy_target: torch.Tensor,
                 obs_pred: torch.Tensor,
                 obs_target: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute auxiliary losses
        
        Args:
            energy_pred: [B, T, 1] predicted energy
            energy_target: [B, T] actual energy
            obs_pred: [B, T, obs_dim] predicted observations
            obs_target: [B, T, obs_dim] actual observations
            mask: [B, T] mask for valid steps
        
        Returns:
            total auxiliary loss
        """
        # Energy prediction loss
        energy_loss = F.mse_loss(energy_pred.squeeze(-1), energy_target)

        obs_loss = self.obs_criterion(obs_pred, obs_target)
        
        ## Observation prediction loss
        #if self.obs_prediction_type == 'classification':
        #    # For classification, obs_pred is [B, T, obs_dim, num_classes]
        #    B, T, obs_dim, num_classes = obs_pred.shape
        #    obs_pred = obs_pred.view(B * T * obs_dim, num_classes)
        #    obs_target = obs_target.view(B * T * obs_dim).long()
        #    obs_loss = self.obs_criterion(obs_pred, obs_target)
        #else:
        #    # For regression, both are [B, T, obs_dim]
        #    obs_loss = self.obs_criterion(obs_pred, obs_target)
        
        # Apply mask if provided
        if mask is not None:
            valid_ratio = mask.sum() / (mask.numel() + 1e-8)
            energy_loss = energy_loss * valid_ratio
            obs_loss = obs_loss * valid_ratio
        
        # Weighted sum
        total_loss = (self.energy_coef * energy_loss + self.obs_coef * obs_loss)
        return total_loss


class ValueLoss:
    """Value function loss"""
    
    def __init__(self, value_coef: float = 0.5):
        self.value_coef = value_coef
        self.criterion = nn.MSELoss()
    
    def __call__(self,
                 value_pred: torch.Tensor,
                 returns: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute value loss
        
        Args:
            value_pred: [B, T, 1] predicted values
            returns: [B, T] actual returns
            mask: [B, T] mask for valid steps
        
        Returns:
            value loss
        """
        value_loss = self.criterion(value_pred.squeeze(-1), returns)
        
        if mask is not None:
            valid_ratio = mask.sum() / (mask.numel() + 1e-8)
            value_loss = value_loss * valid_ratio
        
        return self.value_coef * value_loss


class CompositeLoss:
    """Composite loss combining policy, value, and auxiliary losses"""
    
    def __init__(self,
                 policy_loss: PolicyLoss,
                 auxiliary_loss: Optional[AuxiliaryLoss] = None,
                 value_loss: Optional[ValueLoss] = None):
        self.policy_loss = policy_loss
        self.auxiliary_loss = auxiliary_loss
        self.value_loss = value_loss
    
    def __call__(self,
                 logits: torch.Tensor,
                 value_pred: Optional[torch.Tensor],
                 aux_pred: Optional[Tuple[torch.Tensor, torch.Tensor]],
                 actions: torch.Tensor,
                 rewards: torch.Tensor,
                 values: Optional[torch.Tensor],
                 aux_targets: Optional[Tuple[torch.Tensor, torch.Tensor]],
                 mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute composite loss
        
        Returns:
            total_loss: combined loss
            loss_dict: individual loss components
        """
        loss_dict = {}
        
        # Policy loss
        policy_loss, entropy = self.policy_loss(logits, actions, rewards, mask)
        loss_dict['policy_loss'] = policy_loss.item()
        loss_dict['entropy'] = entropy.item()
        
        total_loss = policy_loss
        
        # Value loss
        if self.value_loss is not None and value_pred is not None and values is not None:
            returns = self.policy_loss._compute_returns(rewards, mask)
            value_loss = self.value_loss(value_pred, returns, mask)
            loss_dict['value_loss'] = value_loss.item()
            total_loss = total_loss + value_loss
        
        # Auxiliary loss
        if self.auxiliary_loss is not None and aux_pred is not None and aux_targets is not None:
            energy_pred, obs_pred = aux_pred
            energy_target, obs_target = aux_targets
            aux_loss = self.auxiliary_loss(energy_pred, energy_target, obs_pred, obs_target, mask)
            loss_dict['aux_loss'] = aux_loss.item()
            total_loss = total_loss + aux_loss
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict