#!/usr/bin/env bash
# run_dtst.sh — launch training with paper-repro settings

python main.py \
  --train_path    "dataset/all" \
  --val_path      "dataset/all" \
  --train_list    "data/fold1/train.txt" \
  --val_list      "data/fold1/eval.txt" \
  --test_path    dataset/all \
  --test_list      "data/fold1/test.txt" \
  --checkpoint   "/home/ubuntu/astro/checkpoints/epoch=000-val_IQR=155.1062.ckpt" \
  --crop          400 \
  --skip          25 \
  --mask_random   0.3 \
  --block_len     10 \
  --batch_size     64 \
  --num_workers    24 \
  --epochs         0 \
  --seed           0
