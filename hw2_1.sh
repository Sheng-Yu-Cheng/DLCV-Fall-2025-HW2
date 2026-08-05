#!/bin/bash

# python hw2_1/train.py \
#   --data-root hw2_data/digits \
#   --output-dir runs/cfg_digits_multicond \
#   --epochs 10 \
#   --batch-size 8 \
#   --lr 1e-4 \
#   --no-amp \
#   --sample-every 0 \
#   --save-every 1 \
#   --p-uncond 0.1 \
#   --p-drop-digit-only 0.1 \
#   --p-drop-dataset-only 0.1 \
#   --resume runs/cfg_digits_fp32/checkpoints/latest.pth

# python hw2_1/inference.py \
#   --checkpoint runs/cfg_digits_fp32/checkpoints/latest.pth \
#   --output-folder output_folder \
#   --samples-per-digit 50 \
#   --batch-size 8 \
#   --guidance-scale 1.0 \
#   --overwrite

for scale in 1 2 4 8; do
  python hw2_1/diagnose-condition.py \
    --checkpoint runs/cfg_digits_multicond/checkpoints/epoch_0010.pth \
    --output-dir diagnose_digit_scale_${scale} \
    --steps 50 \
    --digit-guidance-scale ${scale} \
    --dataset-guidance-scale 1.0 \
    --seeds 42
done