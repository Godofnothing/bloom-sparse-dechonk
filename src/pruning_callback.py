import torch

from transformers import TrainerCallback


def sparsity(x):
    return (1 - torch.count_nonzero(x) / x.numel()).item()


class PruningCallback(TrainerCallback):

    def __init__(self, pruning_modifier, log_frequency: int = -1) -> None:
        # setup modifier
        self.pruning_modifier = pruning_modifier
        self.log_frequency = log_frequency
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.pruning_modifier.initialize(model)

    def on_step_begin(self, args, state, control, **kwargs):
        self.pruning_modifier.check_mask_update(state.global_step)
        if args.global_step % self.log_frequency and args.local_rank == 0:
            print('-' * 10)
            print('Param sparsities')
            for param_name, param in self.pruning_modifier.params.items():
                print(f'{param_name:>25}: {sparsity(param):.4f}')
    
    def on_step_end(self, args, state, control, **kwargs):
        # mask params
        with torch.no_grad():
            for param_name, param in self.pruning_modifier.params.items():
                param.data *= self.pruning_modifier.masks[param_name]

    def on_train_end(self, args, state, control, **kwargs):
        self.pruning_modifier.finalize()
        