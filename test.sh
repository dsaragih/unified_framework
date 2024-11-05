# srun --partition=gpunodes --mem=8G --gres=gpu:rtx_a6000 \
#     python infer_h5.py \
#     --ckpt c2b_optimal.pth \
#     --gpu 0 \
#     --save_gif \
#     --two_bucket

srun --partition=debugnodes --mem=8G --gres=gpu:nvidia_titan_v \
    python test.py \
    --ckpt "model_000004.pth" \
    --gpu 0 \
    --two_bucket \
    --blocksize 2 \
    --subframes 4 \
    --mask_path "./data/2x2_mask.mat"