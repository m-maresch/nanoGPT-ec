# train a character-level openwebtext model
# based on the GPT-2 config

out_dir = "out-openwebtext-char"
eval_interval = 50
eval_iters = 20
log_interval = 1

# only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = "owt"
wandb_run_name = "gpt2-124M"

# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 9
batch_size = 12
block_size = 64
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 1  # override this
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 6e-5
