# train a character-level openwebtext model
# based on the GPT-2 config
compile = False
max_iters = 10
dtype = "float32"

out_dir = "out-openwebtext-char-medium"
eval_interval = 5
eval_iters = 3
log_interval = 1

# only save when val improves
always_save_checkpoint = False

wandb_log = False

# data
dataset = "openwebtext"
batch_size = 32
block_size = 384
# model
n_layer = 14
n_head = 8
n_embd = 384
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 6e-5
