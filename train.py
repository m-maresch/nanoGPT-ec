"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with data parallelism.

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with data parallelism on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with data parallelism on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import logging
import os
import time
import math
import pickle
import sys
from contextlib import nullcontext

import numpy as np
import torch

from torch.distributed.optim import ZeroRedundancyOptimizer

from model import GPTConfig, GPT, compute_loss, configure_optimizers

import loftnn

from loftnn import DataParallel, HybridPipelineParallel, PipelineParallel
from loftnn.configuration import HybridPipelinePlanningAlgorithm
from loftnn.types import Device, Worker
from loftnn.worker_utils import free_memory, initially_free_memory, total_memory

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
log_level = "WARNING"
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "openwebtext"
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = (
    "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
# parallelism
parallelism = "hybrid"  # None, 'data', 'pipeline' or 'hybrid'
# pipeline parallelism
num_microbatches = 4
# hybrid pipeline parallelism
planner = HybridPipelinePlanningAlgorithm.exact
compute_capacities = []
batch_size_limits = []
split_points = []
device_groups = []
samples_allocated = []
activation_checkpointing_budgets = []
use_ambp = True
# other
plot = False
run_experiments = False

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str, list))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

logging.basicConfig(level=log_level)

is_data_parallel = parallelism == "data"
is_pipeline_parallel = parallelism == "pipeline"
is_hybrid_pipeline_parallel = parallelism == "hybrid"

has_pipeline_parallelism = is_pipeline_parallel or is_hybrid_pipeline_parallel

if loftnn.is_available():
    print("LoftNN is available")
    process_config = loftnn.ProcessConfiguration.from_env()
    master_process = (
        process_config.rank == 0
    )  # this process will do logging, checkpointing etc.

    seed_offset = 0
    if is_data_parallel:
        seed_offset = process_config.rank  # each process gets a different seed

    if gradient_accumulation_steps > 1:
        assert (
            not has_pipeline_parallelism
        ), "misconfiguration: gradient accumulation used with pipeline parallelism"
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % process_config.world_size == 0
        gradient_accumulation_steps //= process_config.world_size
else:
    print("LoftNN is not available")
    # if not data parallel, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    process_config = loftnn.ProcessConfiguration(rank=0, local_rank=0, world_size=1)
tokens_per_iter = (
    gradient_accumulation_steps * process_config.world_size * batch_size * block_size
)
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)


def seed():
    torch.manual_seed(1337 + seed_offset)


seed()

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    # torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# poor man's data loader
data_dir = os.path.join("data", dataset)


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
    )
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

X, Y = get_batch("train")  # fetch the very first batch

if plot and master_process:
    from loftnn.tools.plot import plot_model, plot_parameters_by_layer

    plot_model(model, "gpt", X, Y)
    plot_parameters_by_layer(model)

if run_experiments:
    from loftnn.experiments import ExperimentRunner
    from loftnn.worker_groups import randomized_worker_group

    import loftnn.experiment_planning_time_scaling

    configurations = [
        (
            f"GPT with {i} workers",
            model,
            torch.Size([320, block_size]),
            X.dtype,
            randomized_worker_group(i, batch_size_limit=320),
            320,
            4,
            False,
            Device.cuda if device_type == "cuda" else Device.cpu,
        )
        for i in range(3, 10 + 1)
    ]
    ExperimentRunner.run_all(configurations)

    sys.exit()

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


def loss_fn(logits, targets):
    logits = logits.to(device)
    loss = compute_loss(logits, targets)
    return scaler.scale(loss)


if is_data_parallel:
    dist_model = DataParallel(
        model=model,
        process_config=process_config,
        device=Device.cuda if device_type == "cuda" else Device.cpu,
    )
elif is_pipeline_parallel:
    microbatch_sample = X.chunk(num_microbatches)[0]
    pipeline_config = loftnn.PipelineConfiguration(
        split_points=split_points,
        num_microbatches=num_microbatches,
        microbatch_sample=microbatch_sample,
        loss_fn=loss_fn,
    )

    dist_model = PipelineParallel(
        model,
        process_config=process_config,
        pipeline_config=pipeline_config,
        device=Device.cuda if device_type == "cuda" else Device.cpu,
    )
elif is_hybrid_pipeline_parallel:
    microbatch_sample = X.chunk(num_microbatches)[0]
    pipeline_config = loftnn.HybridPipelineConfiguration(
        planner=planner,
        num_microbatches=num_microbatches,
        microbatch_sample=microbatch_sample,
        loss_fn=loss_fn,
    )

    dist_model = HybridPipelineParallel(
        model,
        process_config=process_config,
        hybrid_pipeline_config=pipeline_config,
        device=Device.cuda if device_type == "cuda" else Device.cpu,
    )

    if split_points and device_groups and samples_allocated:
        device_groups_complete = [0] + device_groups + [process_config.world_size]
        samples_allocated = [
            {
                Worker(
                    rank=r,
                    compute_capacity=(
                        compute_capacities[r] if compute_capacities else 1 / 1000
                    ),
                    batch_size_limits=(
                        batch_size_limits[r] if batch_size_limits else 100
                    ),
                ): samples_allocated[r]
                for r in range(device_groups_complete[g], device_groups_complete[g + 1])
            }
            for g in range(len(device_groups_complete) - 1)
        ]
    else:
        (
            split_points,
            device_groups,
            samples_allocated,
            activation_checkpointing_budgets,
        ) = dist_model.compute_plan(use_ambp, batch_size_limits)

    print("using hybrid pipeline parallelism with the following plan:")
    print(f"    split points = {split_points}")
    print(f"    device groups = {device_groups}")
    print(f"    samples allocated = {samples_allocated}")
    print(f"    activation checkpointing budgets = {activation_checkpointing_budgets}")

    dist_model.prepare_schedule(
        split_points, device_groups, samples_allocated, activation_checkpointing_budgets
    )

    # computing the plan advances the RNG of the master process
    seed()  # needed to make sure that X and Y are aligned during training
else:
    dist_model = model

no_split_or_last_process = (
    not has_pipeline_parallelism or process_config.rank == process_config.world_size - 1
)

master_or_last_process = (master_process and not has_pipeline_parallelism) or (
    has_pipeline_parallelism and process_config.rank == process_config.world_size - 1
)

raw_model = model
running_mfu = -1.0

# optimizer
optimizer = configure_optimizers(
    dist_model.named_parameters(),
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type,
)

if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory


# helps estimate an arbitrarily accurate loss over either split using many batches
def estimate_loss():
    out = {}

    with nullcontext() if has_pipeline_parallelism else torch.no_grad():
        dist_model.eval()

        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = dist_model(X, Y)
                if no_split_or_last_process:
                    losses[k] = loss.item()
            out[split] = losses.mean()

        dist_model.train()

    return out


def evaluate():
    global best_val_loss

    losses = estimate_loss()

    optimizer_state = None
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer.consolidate_state_dict(to=process_config.world_size - 1)
        if master_or_last_process:
            optimizer_state = optimizer.state_dict()
    else:
        optimizer_state = optimizer.state_dict()
    checkpoint = {
        "model": dist_model.state_dict(),
        "optimizer": optimizer_state,
        "model_args": model_args,
        "iter_num": iter_num,
        "config": config,
    }

    if master_or_last_process:
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            checkpoint["best_val_loss"] = best_val_loss
            if iter_num > 0 and not has_pipeline_parallelism:
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if has_pipeline_parallelism:
        torch.save(
            checkpoint,
            os.path.join(out_dir, f"ckpt-rank-{process_config.rank}.pt"),
        )


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
tstart = time.time()
t0 = time.time()
dts = []
losses = []
local_iter_num = 0  # number of iterations in the lifetime of this process
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num > 0 and iter_num % eval_interval == 0:
        evaluate()
        t0 = time.time()

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if is_data_parallel:
            # in data parallel training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = dist_model(X, Y)

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")

        if no_split_or_last_process:
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation

        if not has_pipeline_parallelism:
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(dist_model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    if local_iter_num == 0:
        print(f"memory usage: {total_memory(device) - free_memory(device):,}")
        print(
            f"    of training: {initially_free_memory(device) - free_memory(device):,}"
        )

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    dts.append(dt)
    t0 = t1
    if iter_num % log_interval == 0 and master_or_last_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        losses.append(lossf)

        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num >= max_iters:
        break

tend = time.time()
print(f"Training ran for {(tend - tstart):.2f} seconds")
print(f"Iter times: {dts} seconds")
print(f"Losses: {losses}")

if loftnn.is_available():
    dist_model.cleanup()
