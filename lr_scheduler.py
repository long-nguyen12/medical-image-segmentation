import math
from typing import Any

class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, iters_per_epoch=0, lr_step=0, warmup_epochs=0) -> None:
        self.lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch) -> Any:
        T = epoch * self.iters_per_epoch + i
        lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)
        
    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            for i in range(len(optimizer.param_groups)):
                if optimizer.param_groups[i]['lr'] > 0:
                    optimizer.param_groups[i]['lr'] = lr
