#!/bin/bash

which python3

# python hw2_1/train.py \
#   --data-root hw2_data/digits \
#   --output-dir runs/cfg_joint_class \
#   --epochs 10 \
#   --batch-size 8 \
#   --lr 1e-4 \
#   --no-amp \
#   --sample-every 0 \
#   --save-every 1

python3 hw2_1/inference.py \
  --checkpoint runs/cfg_joint_class/checkpoints/epoch_0010.pth \
  --output-folder hw2_1/output \
  --samples-per-digit 50 \
  --batch-size 32 \
  --steps 50 \
  --guidance-scale 2.0

# python hw2_1/diagnose-condition.py \
#   --checkpoint runs/cfg_joint_class/checkpoints/epoch_0010.pth \
#   --output-dir diagnose_joint_epoch10 \
#   --steps 50 \
#   --guidance-scale 2.0 \
#   --seeds 42 43 44

python3 hw2_1/digit_classifier.py --folder hw2_1/output --checkpoint hw2_1/Classifier.pth