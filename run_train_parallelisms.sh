MAX_ITERS=20

echo $'\n============================== No parallelism ==============================\n'
python train.py config/train_openwebtext_char.py --device=cpu --compile=False --max_iters=$MAX_ITERS --parallelism='none'

echo $'\n============================== Data parallelism ==============================\n'
torchrun --nproc_per_node=3 train.py config/train_openwebtext_char.py --device=cpu --compile=False --max_iters=$MAX_ITERS --parallelism="data"

echo $'\n============================== Pipeline parallelism ==============================\n'
torchrun --nproc_per_node=3 train.py config/train_openwebtext_char.py --device=cpu --compile=False --max_iters=$MAX_ITERS --parallelism="pipeline"
