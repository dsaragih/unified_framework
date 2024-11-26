#!/bin/bash

# srun --partition=gpunodes --mem=8G --gres=gpu:rtx_a6000 \
#     python infer_h5.py \
#     --ckpt c2b_optimal.pth \
#     --gpu 0 \
#     --save_gif \
#     --two_bucket


srun --partition=debugnodes --mem=8G --nodelist=tensor1 --gres=gpu:nvidia_titan_v \
    python test.py \
    --ckpt "model_6x6_000219.pth" \
    --gpu 0 \
    --two_bucket \
    --blocksize 6 \
    --subframes 36 \
    --mask_path "./data/6x6_mask.mat"