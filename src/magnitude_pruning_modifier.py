import re
import torch
import wandb

from functools import partial


def mask_gradient_hook(grad, mask):
    return mask * grad

def polynomial_inter(step, pow, init_value, final_value, init_step, final_step):
    assert init_step < final_step, "init step has to be smaller than the final one"
    step = min(max(step, init_step), final_step)
    return init_value + (final_value - init_value) * ((step - init_step) / (final_step - init_step)) ** (1 / pow)


class MagnitudePruningModifier:

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_step: int,
        end_step: int,
        update_frequency: int,
        prunable_params: str = "__ALL__",
        comp_scores_on_cpu: bool = False,
        global_sparsity: bool = False,
        inter_pow: float = 3.0,
        log_wandb: bool = False
    ):
        # pruning schedule
        self.start_step = start_step
        self.end_step = end_step
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        self.update_frequency = update_frequency
        self.prunable_params = prunable_params
        self.comp_scores_on_cpu = comp_scores_on_cpu
        self.global_sparsity = global_sparsity
        self.inter_pow = inter_pow
        self.log_wandb = log_wandb
        self.masks  = {}
        self.params = {}
        # set current sparsity
        self.current_sparsity = 0.0

    def initialize(self, model):
        for param_name, param in model.named_parameters():
            if self.prunable_params == "__ALL__" or re.search(self.prunable_params, param_name):
                mask = torch.ones(param.shape, dtype=torch.bool, device=param.device)

                self.masks[param_name]  = mask
                self.params[param_name] = param

    def check_mask_update(self, step: int) -> bool:
        if (step - self.start_step) % self.update_frequency > 0:
            return False
        else:
            self.set_current_sparsity(step)
            if self.log_wandb:
                wandb.log({'sparsity': self.current_sparsity}, step=step)
            self.mask_update()
            return True

    def set_current_sparsity(self, epoch):
        self.current_sparsity = polynomial_inter(
            epoch, 
            self.inter_pow,
            self.init_sparsity,
            self.final_sparsity,
            self.start_step,
            self.end_step
        )

    @torch.no_grad()
    def mask_update(self):
        # do nothing for 0 sparsity
        if self.current_sparsity == 0:
            return

        if self.global_sparsity:
            scores = []
            for param_name, param in self.params.items():
                score_device = 'cpu' if self.comp_scores_on_cpu else param.device
                scores.append(param.abs().view(-1).to(score_device))
            scores = torch.cat(scores)
            threshold, _ = torch.kthvalue(scores, k=int(len(scores) * self.current_sparsity))
            for param_name, param in self.params.items():
                mask = (param.abs() > threshold)
                param.data = param * mask

                self.masks[param_name] = mask
        else:
            for param_name, param in self.params.items():
                scores = param.abs().view(-1)
                threshold, _ = torch.kthvalue(scores, k=int(len(scores) * self.current_sparsity))

                mask = param.abs() > threshold
                param.data = param * mask

                self.masks[param_name] = mask

    def finalize(self):
        del self.masks
