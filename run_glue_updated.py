import logging
import os
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import datasets
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils.versions import require_version
from adalora.adalora import RankAllocator 
import json

# SoRA specific dependencies
from src.trainer import SparseTrainer 
from src.util import compute_trainable_sparse_param, create_optimizer_and_scheduler
from src.sparse_optimizer import SparseAdamW
from transformers import get_linear_schedule_with_warmup

# Adalora specific dependencies
from adalora.trainer import AdaTrainer

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_best_metric = {
    "rte": "eval_accuracy",
    "mrpc": "eval_f1", 
    "cola": "eval_matthews_correlation", 
    "stsb": "eval_pearson", 
    "sst2": "eval_accuracy", 
    "qnli": "eval_accuracy",
    "mnli": "eval_accuracy",
    "mnli-m": "eval_accuracy",
    "mnli-mm": "eval_accuracy",
    "qqp": "eval_accuracy",
}

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    # cls_dropout: Optional[float] = field(default=None)
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    lora_method: str = field(
        default="lora",
        metadata={"help": "Lora method(lora,sora,adalora)"},
    )
    # Sora specific arguments
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    # LoRA specific arguments
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )
    # Adalora specific arguments
    lora_type: Optional[str] = field(
        default="frd",
        metadata={"help": "The lora type: frd or svd."},
    )
    lora_module: Optional[str] = field(
        default="query,value",
        metadata={"help": "The modules applying lora: query,key,value,intermediate,layer.output,attention.output"},
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default='houlsby',
        metadata={"help": "houlsby or pfeiffer"},
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    apply_adalora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply rank selector or not."},
    )
    target_rank: Optional[int] = field(
        default=16,
        metadata={"help": "Average target rank."},
    )
    target_total_rank: Optional[int] = field(
        default=None,
        metadata={"help": "Specifying target number of total singular values"},
    )
    init_warmup: Optional[int] = field(
        default=4500,
        metadata={"help": "Total steps of inital warmup"},
    )
    final_warmup: Optional[int] = field(
        default=12000,
        metadata={"help": "Total steps of final fine-tuning"},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    reg_orth_coef: Optional[float] = field(
        default=0.0,
        metadata={"help": "Orthogonal regularization coefficient"},
    )
    mask_interval: Optional[int] = field(
        default=10,
        metadata={"help": "Masking interval"},
    )
    beta1: Optional[float] = field(
        default=0.85,
        metadata={"help": "The coefficient of EMA"},
    )
    beta2: Optional[float] = field(
        default=0.85,
        metadata={"help": "The coefficient of EMA"},
    )

# SoRA specific dependencies
@dataclass
class SparseArguments:
    sparse_lambda: Optional[float] = field(
        default=1e-3, metadata={"help": "loss penalty term for gate param"}
    )
    sparse_lambda_2: Optional[float] = field(
        default=1e-3, metadata={"help": "clipping scale for gate param"}
    )
    sparse_lr: Optional[float] = field(
        default=None, metadata={"help": "lr for gate parameter in sparse lora, default to same as learning rate for other parameters"}
    )
    sparse_lora_r: Optional[int] = field(
        default=16, metadata={"help": "matrix rank in lora"}
    )
    lambda_schedule: Optional[str] = field(
        default=None, metadata={"help": "scheduling of lambda_2, {linear, log_linear}"}
    )
    max_lambda: Optional[float] = field(
        default=10, metadata={"help": "maximum value of lambda_2 in scheduling"}
    )
    lambda_num: Optional[int] = field(
        default=10, metadata={"help": "total number of lambdas in scheduling"}
    )
    # cls_dropout: Optional[float]= field(default=0.0)
    
# SoRA specific dependencies
@dataclass
class SparseTrainingArguments(TrainingArguments):
    train_sparse: Optional[bool] = field(
        default=False, metadata={"help": "whether use sparse lora"}
    )
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SparseTrainingArguments, SparseArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, sparse_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, sparse_args = parser.parse_args_into_dataclasses()
    training_args.report_to=[]
    # training_args.eval_accumulation_steps = 2
    task_name_for_get = data_args.task_name
    if "mnli" in data_args.task_name:
        data_args.task_name = "mnli"

    if model_args.lora_method == "sora":
        training_args.metric_for_best_model = task_to_best_metric[data_args.task_name]

    # Rank of the process during distributed training (GPU index) 
    if os.getenv("LOCAL_RANK"):
        training_args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        training_args.local_rank = -1
    
    # setting sparse learning rate
    if training_args.train_sparse:
        if sparse_args.sparse_lr is None:
            sparse_args.sparse_lr = training_args.learning_rate

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "nyu-mll/glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    # print(raw_datasets)

    if model_args.lora_method == "lora":
        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            # cls_dropout=training_args.cls_dropout,
            apply_lora=model_args.apply_lora,
            lora_alpha=model_args.lora_alpha,
            lora_r=model_args.lora_r,
            apply_adapter=model_args.apply_adapter,
            adapter_type=model_args.adapter_type,
            adapter_size=model_args.adapter_size,
            reg_loss_wgt=model_args.reg_loss_wgt,
            masking_prob=model_args.masking_prob,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        trainable_params = []
        if model_args.apply_lora:
            if model_args.lora_path is not None:
                lora_state_dict = torch.load(model_args.lora_path)
                logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
                logger.info(lora_state_dict.keys())
                model.load_state_dict(lora_state_dict, strict=False)
            trainable_params.append('lora')
        
        if len(trainable_params) > 0:
            for name, param in model.named_parameters():
                print(name)
                if name.startswith('deberta') or name.startswith('roberta'):
                    param.requires_grad = False
                    for trainable_param in trainable_params:
                        if trainable_param in name:
                            param.requires_grad = True
                            break
                else:
                    param.requires_grad = True
    
    elif model_args.lora_method == "sora":
        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

        if training_args.train_sparse:
            print("loading from src.lora")
            from src.lora import LoraModel, LoraConfig
        else:
            from opendelta.delta_models import LoraModel, LoraConfig

        lora_config = json.load(open("config/lora_config.json"))
        lora_config["lora_r"] = sparse_args.sparse_lora_r
        lora_config = LoraConfig.from_dict(lora_config)
        delta_model = LoraModel.from_config(lora_config, backbone_model=model)
        delta_model.freeze_module(set_state_dict = True)
        delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=False)
    elif model_args.lora_method == "adalora":
        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            # cls_dropout=training_args.cls_dropout,
            apply_lora=model_args.apply_lora,
            lora_type=model_args.lora_type, 
            lora_module=model_args.lora_module, 
            lora_alpha=model_args.lora_alpha,
            lora_r=model_args.lora_r,
            apply_adapter=model_args.apply_adapter,
            adapter_type=model_args.adapter_type,
            adapter_size=model_args.adapter_size,
            reg_loss_wgt=model_args.reg_loss_wgt,
            masking_prob=model_args.masking_prob,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        trainable_params = []
        if model_args.apply_lora:
            if model_args.lora_path is not None:
                lora_state_dict = torch.load(model_args.lora_path)
                logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
                logger.info(lora_state_dict.keys())
                model.load_state_dict(lora_state_dict, strict=False)
            trainable_params.append('lora')

        if model_args.apply_adapter:
            if model_args.adapter_path is not None:
                adapter_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_adapter.bin'))
                head_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_model_head.bin'))
                added_state_dict = {}
                for k, v in adapter_state_dict.items():
                    new_k = k.replace(data_args.task_name + '.', '').replace('adapter_down.0.', 'adapter_A.').replace('adapter_up.', 'adapter_B.').replace('.adapters.', '.adapter.')
                    added_state_dict[new_k] = v
                for k, v in head_state_dict.items():
                    new_k = k.replace('heads.' + data_args.task_name + '.1', 'classifier.dense').replace('heads.' + data_args.task_name + '.4', 'classifier.out_proj')
                    added_state_dict[new_k] = v
                logger.info(f"Apply adapter state dict from {model_args.adapter_path}.")
                logger.info(added_state_dict.keys())
                missing_keys, unexpected_keys = model.load_state_dict(added_state_dict, strict=False)
                for missing_key in missing_keys:
                    assert 'adapter' not in missing_key, missing_key + ' is missed in the model'
                assert len(unexpected_keys) == 0, 'Unexpected keys ' + str(unexpected_keys)
            trainable_params.append('adapter')

        if model_args.apply_bitfit:
            trainable_params.append('bias')

        num_param = 0 
        if len(trainable_params) > 0:
            for name, param in model.named_parameters():
                if name.startswith('deberta') or name.startswith('roberta'):
                    param.requires_grad = False
                    for trainable_param in trainable_params:
                        if trainable_param in name:
                            param.requires_grad = True
                            sub_num_param = 1 
                            for dim in param.shape:
                                sub_num_param *= dim  
                            num_param += sub_num_param 
                            break
                else:
                    param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                sub_num_param = 1 
                for dim in param.shape:
                    sub_num_param *= dim  
                num_param += sub_num_param
        logger.info("Number of Trainable Parameters: %d"%(int(num_param))) 
    else:
        pass

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    
    # prepare train dataset
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    # prepare evaluation dataset
    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_val_samples is not None:
            max_val_samples = min(len(eval_dataset), data_args.max_val_samples)
            eval_dataset = eval_dataset.select(range(max_val_samples))

    # prepare prediction dataset
    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # print(metric)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    if model_args.lora_method == "lora" or model_args.lora_method == "adalora":
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            if data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
    elif model_args.lora_method == "sora":
        def compute_metrics(mode, p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            if data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    if model_args.lora_method == "lora":
        # print("TRAINING ARGUMENTS")
        # print(training_args)
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    elif model_args.lora_method == "sora":
        # Initialize our Trainer
        optimizer, lr_scheduler = create_optimizer_and_scheduler(training_args, model, num_training_steps=int(training_args.num_train_epochs*(len(train_dataset) / training_args.train_batch_size)))
        sparse_optimizer = None
        sparse_scheduler = None
        if training_args.train_sparse:
            print("building sparse optimizer and scheduler")
            from src.trainer import GATE_PARAM_NAME
            valid_param_name = []
            for n, p in model.named_parameters():
                print(n)
                if GATE_PARAM_NAME in n:
                    valid_param_name.append(n)
            print("valid param name:", valid_param_name)
            sparse_optimizer = SparseAdamW(sparse_lambda=sparse_args.sparse_lambda_2, lambda_schedule=sparse_args.lambda_schedule, max_lambda=sparse_args.max_lambda, lambda_num=sparse_args.lambda_num, params=[p for n, p in model.named_parameters() if GATE_PARAM_NAME in n and p.requires_grad], lr=sparse_args.sparse_lr)
            sparse_scheduler = get_linear_schedule_with_warmup(sparse_optimizer, 
            num_warmup_steps=int(training_args.num_train_epochs*(len(train_dataset) / training_args.train_batch_size)*training_args.warmup_ratio), 
            num_training_steps=int(training_args.num_train_epochs*(len(train_dataset) / training_args.train_batch_size)))
        
        # Initialize our Trainer
        trainer = SparseTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            optimizers = (optimizer, lr_scheduler),
            sparse_lambda = sparse_args.sparse_lambda,
            sparse_optimizer = (sparse_optimizer, sparse_scheduler)
        )
    elif model_args.lora_method == "adalora":
        # Initialize the rankallocator
        if model_args.lora_type == "svd" and model_args.apply_adalora:
            rankallocator = RankAllocator(
                model, 
                lora_r=model_args.lora_r,
                target_rank=model_args.target_rank,
                init_warmup=model_args.init_warmup, 
                final_warmup=model_args.final_warmup,
                mask_interval=model_args.mask_interval, 
                beta1=model_args.beta1, 
                beta2=model_args.beta2, 
                target_total_rank=model_args.target_total_rank, 
            )
        else:
            rankallocator = None

        # Initialize our Trainer
        trainer = AdaTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            rankallocator=rankallocator,
            model_args=model_args,  
        )

    else:
        pass
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        # trainer.save_state()

    if model_args.lora_method == "sora":
        sparse_param, total_param = compute_trainable_sparse_param(model)
        # eval on n samples train set
        train_dataset_for_eval = train_dataset.shuffle(seed=42).select(range(2))
        logger.info("*** Evaluate on training subset ***")
        metrics = trainer.evaluate(eval_dataset=train_dataset_for_eval, metric_key_prefix = "eval_train")
        # print("METTRICS",metrics)
        trainer.log_metrics("eval_train", metrics)
        trainer.save_metrics("eval_train", metrics)
        BEST_TRAIN_METRIC = metrics["eval_train_" + "_".join(task_to_best_metric[data_args.task_name].split("_")[1:])]

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            if model_args.lora_method == "sora":
                BEST_EVAL_METRIC = metrics[task_to_best_metric[data_args.task_name]]

    # Prediction
    if model_args.lora_method == "lora" or model_args.lora_method=="adalora":
        if training_args.do_predict:
            logger.info("*** Test ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [data_args.task_name]
            predict_datasets = [predict_dataset]
            if data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                predict_datasets.append(raw_datasets["test_mismatched"])

            for predict_dataset, task in zip(predict_datasets, tasks):
                # Removing the `label` columns because it contains -1 and Trainer won't like that.
                # predict_dataset.remove_columns_("label")
                predictions = trainer.predict(test_dataset=predict_dataset).predictions
                predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

                output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
                if trainer.is_world_process_zero():
                    with open(output_test_file, "w") as writer:
                        logger.info(f"***** Test results {task} *****")
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            if is_regression:
                                writer.write(f"{index}\t{item:3.3f}\n")
                            else:
                                item = label_list[item]
                                writer.write(f"{index}\t{item}\n")
        if model_args.lora_method == "adalora":
            if rankallocator is not None and is_main_process(training_args.local_rank):
                rank_pattern = rankallocator.get_rank_pattern()
                with open(os.path.join(training_args.root_output_dir, "rank_pattern.json"), "w") as f:
                    json.dump(rank_pattern, f) 

    elif model_args.lora_method == "sora":
        if training_args.do_predict:
            logger.info("*** Predict ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [data_args.task_name]
            predict_datasets = [predict_dataset]
            if data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                predict_datasets.append(raw_datasets["test_mismatched"])

            for predict_dataset, task in zip(predict_datasets, tasks):
                metrics = trainer.evaluate(eval_dataset=predict_dataset)

                max_val_samples = (
                    data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
                )
                metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

                trainer.log_metrics("test", metrics)
            
                trainer.save_metrics("test", metrics)
        
        logger.info("***** Final Model ******\nLora rank: %d\nNumber of trainable full param: %d\nNumber of trainable sparse param: %d, Ratio: %.4f%%\n**********" % (lora_config.lora_r, total_param, sparse_param, sparse_param / total_param * 100))

    else:
        pass

    if model_args.lora_method == "sora":
        def compute_metrics_in_schedule(mode, p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            if data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                if mode == "eval":
                    result["generalization"] = result["_".join(task_to_best_metric[data_args.task_name].split("_")[1:])] / BEST_EVAL_METRIC * 100
                elif mode == "eval_train":
                    result["memorization"] = result["_".join(task_to_best_metric[data_args.task_name].split("_")[1:])] / BEST_TRAIN_METRIC * 100
                elif mode == "test":
                    pass
                else:
                    raise NotImplementedError
                return result
            elif is_regression:
                raise NotImplementedError
                
            else:
                raise NotImplementedError
                

        # schedule
        if sparse_args.lambda_schedule is not None:
            logger.info("*****Start lambda_2 scheduling***")
            for _ in range(sparse_args.lambda_num - 1):
                training_args.load_best_model_at_end = False
                sparse_optimizer.step_lambda()
                trainer = SparseTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset if training_args.do_train else None,
                    eval_dataset=[eval_dataset if training_args.do_eval else None, train_dataset_for_eval],
                    compute_metrics=compute_metrics_in_schedule,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    optimizers = (optimizer, lr_scheduler),
                    sparse_lambda = sparse_args.sparse_lambda,
                    sparse_optimizer = (sparse_optimizer, sparse_scheduler),
                )
                
                trainer.train()

                if training_args.do_predict:
                    logger.info("*** Predict ***")

                    # Loop to handle MNLI double evaluation (matched, mis-matched)
                    tasks = [data_args.task_name]
                    predict_datasets = [predict_dataset]
                    

                    for predict_dataset, task in zip(predict_datasets, tasks):
                        metrics = trainer.evaluate(eval_dataset=predict_dataset, metric_key_prefix="test")

                        max_eval_samples = (
                            data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
                        )
                        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))


                        trainer.log_metrics("test", metrics)

                        trainer.save_metrics("test", metrics)

                
                sparse_param, total_param = compute_trainable_sparse_param(model)

                logger.info("***** Lambda=%f Final Model ******\nLora rank: %d\nNumber of trainable full param: %d\nNumber of trainable sparse param: %d, Ratio: %.4f%%\n**********" % (sparse_optimizer.sparse_lambda, lora_config.lora_r, total_param, sparse_param, sparse_param / total_param * 100))


if __name__ == "__main__":
    main()