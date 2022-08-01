from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    tokenizer_name: str = field(metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    model_name: Optional[str] = field(default=None, metadata={
        "help": "Initial model weights: either a name for HF hub or a local path. Specify either this or config_name"})
    config_name: Optional[str] = field(default=None, metadata={
        "help": "Model config: either a name for HF hub or a path to a local .json. Specify either this or model_name"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: str = field(metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    block_size: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    eval_subset_size: int = 512  # how many sequences will be processed in each eval
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

@dataclass
class SparsificationArguments:
    '''
    Arguments specifying the parameters of the pruning modifier.
    '''
    init_sparsity:  float = field(
        metadata={"help": "Init sparsity."},
    )
    final_sparsity: float  = field(
        metadata={"help": "Final sparsity."},
    )
    start_step: Optional[int] = field(
        default=0, 
        metadata={"help": "Start epoch of the pruning procedure."},
    )
    end_step: Optional[int] = field(
        default=-1, 
        metadata={"help": "End epoch of pruning."},
    )
    update_frequency: int = field(
        default=100, 
        metadata={"help": "Frequency of pruning updates."},
    )
    prunable_params: Optional[str] = field(
        default="__ALL__", 
        metadata={"help": "Which params to prune (str, regular expression or __ALL__)"}
    )
    comp_scores_on_cpu: bool = field(
        default=False, 
        metadata={"help": "Whether to compute scores on CPU"}
    )
    global_sparsity: bool = field(
        default=False, 
        metadata={"help": "Whether the sparsity is global"}
    )
    inter_pow: float = field(
        default=3.0, 
        metadata={"help": "Interpolation power in the sparsity schedule"}
    )
    # debug args
    sparsity_log_freq: int = field(
        default=-1, 
        metadata={"help": "Sparsity logging frequency."},
    )