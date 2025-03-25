
from torch.optim.lr_scheduler import LRScheduler
import math
import warnings

class InverseSquareLRWithWarmup(LRScheduler):
    """
    Implements an inverse square learning rate scheduler with warmup steps.
    
    During warmup, the learning rate increases linearly from init_lr to max_lr.
    After warmup, the learning rate decreases according to the inverse square of the step number:
    lr = max_lr * (warmup_steps / step)^2 for step > warmup_steps
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        init_lr (float): Initial learning rate during warmup phase. Default: 0.0
        max_lr (float): Maximum learning rate after warmup phase. Default: 0.1
        warmup_steps (int): Number of warmup steps. Default: 1000
        last_epoch (int): The index of the last epoch. Default: -1
    """
    
    def __init__(self, optimizer, init_lr=0.0, max_lr=0.001, warmup_steps=1000, last_epoch=-1):
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        super(InverseSquareLRWithWarmup, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_steps
            return [self.init_lr + alpha * (self.max_lr - self.init_lr) for _ in self.base_lrs]
        else:
            # Inverse square decay phase
            decay_factor = math.sqrt(self.warmup_steps / self.last_epoch)
            return [self.max_lr * decay_factor for _ in self.base_lrs]
            
    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_steps
            return [self.init_lr + alpha * (self.max_lr - self.init_lr) for _ in self.base_lrs]
        else:
            # Inverse square decay phase
            decay_factor = (self.warmup_steps / self.last_epoch) ** 2
            return [self.max_lr * decay_factor for _ in self.base_lrs]