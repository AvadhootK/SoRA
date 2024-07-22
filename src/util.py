import torch
from transformers.trainer_pt_utils import get_parameter_names
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
GATE_PARAM_NAME= "lora.gate"

# In sparse regularization techniques, certain parameters are forced to be zero. 
# These zero parameters do not contribute to the model's computations or its capacity to learn, 
# thus effectively reducing the model's complexity.

# calculates the number of trainable parameters in a model, accounting for sparsity in certain parameters
# returns: number of trainable parameters after accounting for sparsity,  total number of trainable parameters without considering sparsity
def compute_trainable_sparse_param(model):

    total_trainable_param = 0
    deduct = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            # for parameters associated with sparsity (gate unit)
            if GATE_PARAM_NAME in n:
                # calculate the number of zero elements
                # torch.numel() - return total number of elements in input tensor
                # torch.count_nonzero - Counts the number of non-zero values in the tensor input
                deduct += (torch.numel(p) - torch.count_nonzero(p)) * model.config.hidden_size * 2  # zero_number * 768 * 2
            else:
                total_trainable_param += torch.numel(p)
    sparse_trainable_param = total_trainable_param - deduct
    return sparse_trainable_param, total_trainable_param

# sets up/returns optimizer and scheduler created using helper functions
def create_optimizer_and_scheduler(args, model, num_training_steps: int):
    """
    Setup the optimizer and the learning rate scheduler.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method (or :obj:`create_optimizer`
    and/or :obj:`create_scheduler`) in a subclass.
    """
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, num_training_steps=num_training_steps, optimizer=optimizer)
    return optimizer, scheduler

# Standard Adam optimizer with weight decay (L2 regularization)
# Used to optimize Lora_A and Lora_B matrices in LoRA method
def create_optimizer(args, model):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """

    # organizes model parameters into two groups for optimization: 
    # one with weight decay applied to selected parameters excluding biases and normalization layers, 
    # and another without weight decay applied
    # gate unit is excluded in standard optimization
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    print(f"removing {GATE_PARAM_NAME} from standard optimizer")
    optimizer_grouped_parameters = [
        {   
            # weights of certain layers requiring regularization during optimization
            "params": [p for n, p in model.named_parameters() if n in decay_parameters and GATE_PARAM_NAME not in n and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {   
            # model biases and other parameters that do not need regularization
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters and GATE_PARAM_NAME not in n and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = args.learning_rate
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    return optimizer

# initialize a learning rate scheduler to adjust the learning rate during training
def create_scheduler(args, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    passed as an argument.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps)
    return lr_scheduler
