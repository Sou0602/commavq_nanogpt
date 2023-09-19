out_dir = "out-commavq"
eval_interval = 500
eval_iters = 200
log_interval = 10

always_save_checkpoint = (
    False  ## When False, only updates the model when val loss improves
)

init_from = "scratch"  # change to 'resume' , when training from a checkpoint

wandb_log = False  # override via command line if you like
wandb_project = "commavq"
wandb_run_name = "nano-gpt"

dataset = "commavq"
gradient_accumulation_steps = 1
batch_size = 12  ## based on number of parameters and compute size
block_size = 129 * 20  # 129 tokens per frame times the number of context frames.

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2

learning_rate = 1e-4
max_iters = 500000
lr_decay_iters = 500000
min_lr = 1e-5
# beta2 = 0.99 # make a bit bigger because number of tokens per iter is small, uncomment if the tokens/iteration is small.

warmup_iters = 1000  # not super necessary potentially

## vocab size = 1048
