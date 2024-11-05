srun --partition=gpunodes --mem=8G --gres=gpu:nvidia_titan_rtx \
    python train.py \
        --expt davis \
        --batch 64 \
        --gpu 0,1 \
        --blocksize 2 \
        --subframes 4 \
        --two_bucket \
        --mask 2x2 \
        --ckpt "/u8/d/dsaragih/diffusion-posterior-sampling/unified_framework/models/model_000004.pth"