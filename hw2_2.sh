#!/bin/bash

python3 hw2_2/ddim.py \
    --noise_dir "$1" \
    --output_dir "$2" \
    --model_path "$3"

# python3 hw2_2/ddim.py \
#   --noise_dir ./hw2_data/face/noise \
#   --output_dir ./hw2_2/output \
#   --model_path ./hw2_data/face/UNet.pt