#!/bin/bash

# python hw2_1/train.py \
#   --data-root hw2_data/digits \
#   --output-dir runs/cfg_digits_fp32 \
#   --epochs 50 \
#   --batch-size 8 \
#   --lr 1e-4 \
#   --no-amp \
#   --sample-every 0 \
#   --save-every 1 \
#   --resume runs/cfg_digits_fp32/checkpoints/latest.pth

# python hw2_1/inference.py \
#   --checkpoint runs/cfg_digits_fp32/checkpoints/latest.pth \
#   --output-folder output_folder \
#   --samples-per-digit 50 \
#   --batch-size 8 \
#   --guidance-scale 1.0 \
#   --overwrite

python hw2_1/diagnose-condition.py \
  --checkpoint runs/cfg_digits_fp32/checkpoints/latest.pth \
  --output-dir diagnose_latest \
  --steps 50 \
  --guidance-scale 1.0