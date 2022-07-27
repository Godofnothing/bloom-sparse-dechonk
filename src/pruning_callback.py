from transformers import TrainerCallback


class PruningCallback(TrainerCallback):

    def __init__(self, pruning_modifier) -> None:
        # setup modifier
        self.pruning_modifier = pruning_modifier
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.pruning_modifier.initialize(model)

    def on_step_begin(self, args, state, control, **kwargs):
        self.pruning_modifier.check_mask_update(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        self.pruning_modifier.finalize()
        