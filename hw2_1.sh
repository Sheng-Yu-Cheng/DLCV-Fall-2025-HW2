#!/usr/bin/env bash

set -euo pipefail

which python3

# TA usage:
#   bash hw2_1.sh <output_directory>

if [[ $# -ne 1 ]]; then
    echo "Usage: bash hw2_1.sh <output_directory>" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DIR="$1"
CHECKPOINT="$ROOT_DIR/hw2_1/runs/cfg_joint_class/checkpoints/epoch_0010.pth"
INFERENCE_SCRIPT="$ROOT_DIR/hw2_1/inference.py"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [[ ! -f "$INFERENCE_SCRIPT" ]]; then
    echo "Error: inference script not found: $INFERENCE_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Python: $(command -v python3)"
echo "Checkpoint: $CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"

python3 "$INFERENCE_SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --output-folder "$OUTPUT_DIR" \
    --samples-per-digit 50 \
    --batch-size 32 \
    --steps 50 \
    --guidance-scale 2.5

echo "Image generation completed."