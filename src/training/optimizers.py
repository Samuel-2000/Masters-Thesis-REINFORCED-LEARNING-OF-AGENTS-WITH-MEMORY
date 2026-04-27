"""
Optimizer utilities for training
"""

import torch
import torch.optim as optim
from typing import List, Optional, Union
import math


class GradientClipper:
    """Gradient clipping utility"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def clip(self, parameters: Union[torch.Tensor, List[torch.Tensor]]):
        """Clip gradients"""
        torch.nn.utils.clip_grad_norm_(
            parameters, 
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )
    
    def get_grad_norm(self, parameters: Union[torch.Tensor, List[torch.Tensor]]) -> float:
        """Compute gradient norm"""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        parameters = [p for p in parameters if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        total_norm = 0.0
        for p in parameters:
            param_norm = p.grad.data.norm(self.norm_type)
            total_norm += param_norm.item() ** self.norm_type
        
        return total_norm ** (1.0 / self.norm_type)


class LearningRateScheduler:
    """Learning rate scheduler with various policies"""
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 mode: str = 'cosine',
                 lr_start: float = 0.001,
                 lr_min: float = 1e-6,
                 warmup_steps: int = 1000,
                 decay_steps: int = 10000,
                 decay_rate: float = 0.96):
        """
        Args:
            optimizer: PyTorch optimizer
            mode: 'cosine', 'exponential', 'step', 'plateau', 'constant'
            lr_start: initial learning rate
            lr_min: minimum learning rate
            warmup_steps: number of warmup steps
            decay_steps: decay steps for exponential/step decay
            decay_rate: decay rate for exponential decay
        """
        self.optimizer = optimizer
        self.mode = mode
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience = 0
        self.max_patience = 10
        
        # Set initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_start
    
    def step(self, loss: Optional[float] = None):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.lr_start * (self.step_count / self.warmup_steps)
        else:
            # Main schedule
            effective_step = self.step_count - self.warmup_steps
            
            if self.mode == 'cosine':
                # Cosine decay
                progress = min(effective_step / self.decay_steps, 1.0)
                lr = self.lr_min + 0.5 * (self.lr_start - self.lr_min) * \
                     (1 + math.cos(math.pi * progress))
            
            elif self.mode == 'exponential':
                # Exponential decay
                lr = self.lr_start * (self.decay_rate ** (effective_step / self.decay_steps))
            
            elif self.mode == 'step':
                # Step decay
                lr = self.lr_start * (self.decay_rate ** (effective_step // self.decay_steps))
            
            elif self.mode == 'plateau':
                # Reduce on plateau
                if loss is not None:
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.patience = 0
                    else:
                        self.patience += 1
                    
                    if self.patience >= self.max_patience:
                        lr = self.optimizer.param_groups[0]['lr'] * 0.5
                        self.patience = 0
                    else:
                        return  # No change
                else:
                    return  # Need loss for plateau mode
            
            elif self.mode == 'constant':
                # Constant learning rate
                lr = self.lr_start
            
            else:
                raise ValueError(f"Unknown scheduler mode: {self.mode}")
            
            # Ensure lr doesn't go below minimum
            lr = max(lr, self.lr_min)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def state_dict(self):
        """Get scheduler state"""
        return {
            'step_count': self.step_count,
            'best_loss': self.best_loss,
            'patience': self.patience,
            'lr_start': self.lr_start,
            'lr_min': self.lr_min,
            'mode': self.mode
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.step_count = state_dict.get('step_count', 0)
        self.best_loss = state_dict.get('best_loss', float('inf'))
        self.patience = state_dict.get('patience', 0)
        self.lr_start = state_dict.get('lr_start', self.lr_start)
        self.lr_min = state_dict.get('lr_min', self.lr_min)
        self.mode = state_dict.get('mode', self.mode)


class OptimizerFactory:
    """Factory for creating optimizers"""
    
    @staticmethod
    def create(optimizer_name: str,
               parameters,
               lr: float = 0.001,
               **kwargs) -> optim.Optimizer:
        """
        Create optimizer
        
        Args:
            optimizer_name: 'adam', 'adamw', 'sgd', 'rmsprop'
            parameters: model parameters
            lr: learning rate
            **kwargs: additional optimizer arguments
        
        Returns:
            Optimizer instance
        """
        optimizer_name = optimizer_name.lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                parameters,
                lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=kwargs.get('weight_decay', 0.0)
            )
        
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                parameters,
                lr=lr,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=kwargs.get('weight_decay', 0.01)
            )
        
        elif optimizer_name == 'sgd':
            return optim.SGD(
                parameters,
                lr=lr,
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 0.0),
                nesterov=kwargs.get('nesterov', False)
            )
        
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(
                parameters,
                lr=lr,
                alpha=kwargs.get('alpha', 0.99),
                eps=kwargs.get('eps', 1e-8),
                weight_decay=kwargs.get('weight_decay', 0.0),
                momentum=kwargs.get('momentum', 0.0)
            )
        
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
