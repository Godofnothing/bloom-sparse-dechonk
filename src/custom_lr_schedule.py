import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


__all__ = [
    "get_cyclic_linear_schedule_with_warmup",
    "get_cyclic_cosine_schedule_with_warmup",
    "create_custom_scheduler"
]


def get_cyclic_linear_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_cycle_steps: int,
    last_epoch: int = -1
):
    assert num_warmup_steps <= num_cycle_steps

    def lr_lambda(current_step: int):
        adj_step = current_step % num_cycle_steps
        warmup_lr_mult = min(float(adj_step) / float(max(1.0, num_warmup_steps)), 1.0)
        ratio = (adj_step - num_warmup_steps) / max(num_cycle_steps - num_warmup_steps - 1, 1.0)
        return warmup_lr_mult * (1 - ratio)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cyclic_cosine_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_cycle_steps: int,
    last_epoch: int = -1
):
    assert num_warmup_steps <= num_cycle_steps

    def lr_lambda(current_step: int):
        adj_step = current_step % num_cycle_steps
        warmup_lr_mult = min(float(adj_step) / float(max(1.0, num_warmup_steps)), 1.0)
        ratio = (adj_step - num_warmup_steps) / max(num_cycle_steps - num_warmup_steps - 1, 1.0)
        cos_mult = max(0.0, 0.5 * (1.0 + math.cos(math.pi * ratio)))
        return warmup_lr_mult * cos_mult

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def create_custom_scheduler(
    lr_scheduler_type: str, 
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_cycle_steps: int,
    last_epoch: int = -1
):
    if lr_scheduler_type == 'cyclic_linear':
        return get_cyclic_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_cycle_steps, last_epoch)
    elif lr_scheduler_type == 'cyclic_cosine':
        return get_cyclic_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_cycle_steps, last_epoch)
    else:
        raise ValueError("Unknown schedule")
