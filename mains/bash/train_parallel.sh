CUDA_VISIBLE_DEVICES=0 python3.5 mains/train.py -c configs/config_parallel_0.json    &
P0=$!

CUDA_VISIBLE_DEVICES=1 python3.5 mains/train.py -c configs/config_parallel_1.json    &
P1=$!

wait $P0 $P1