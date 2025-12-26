#!/bin/bash

# Script to convert DeepSpeed checkpoint to model.safetensors
# Usage: ./convert_checkpoint.sh <checkpoint_directory>
# Example: ./convert_checkpoint.sh outputs/train/tau0/checkpoints/000040/

set -e  # Exit on any error

# Check if checkpoint directory is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide the checkpoint directory path"
    echo "Usage: $0 <checkpoint_directory>"
    echo "Example: $0 outputs/train/tau0/checkpoints/000040/"
    exit 1
fi

CHECKPOINT_DIR="$1"

# Validate that the checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory '$CHECKPOINT_DIR' does not exist"
    exit 1
fi

echo "Converting checkpoint in directory: $CHECKPOINT_DIR"

# Step 1: Convert sharded checkpoint to full state dict
echo "Step 1: Converting sharded checkpoint to full state dict..."
python lerobot/scripts/zero_to_fp32.py "$CHECKPOINT_DIR" "$CHECKPOINT_DIR/full_state_dict" --max_shard_size 1000GB

# Step 2: Convert pytorch_model.bin to model.safetensors
echo "Step 2: Converting pytorch_model.bin to model.safetensors..."
python lerobot/scripts/bin_to_safetensors.py "$CHECKPOINT_DIR/full_state_dict/pytorch_model.bin" --output_file "$CHECKPOINT_DIR/model.safetensors"

echo "Conversion completed successfully!"
echo "Model saved as: $CHECKPOINT_DIR/model.safetensors"
