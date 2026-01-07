#!/bin/bash

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to convert DeepSpeed checkpoint to model.safetensors
# Usage: ./convert_checkpoint.sh <checkpoint_directory>
# Example: ./convert_checkpoint.sh outputs/train/pi05/checkpoints/000040/

set -e  # Exit on any error

# Check if checkpoint directory is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide the checkpoint directory path"
    echo "Usage: $0 <checkpoint_directory>"
    echo "Example: $0 outputs/train/pi05/checkpoints/000040/"
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
python src/opentau/scripts/zero_to_fp32.py "$CHECKPOINT_DIR" "$CHECKPOINT_DIR/full_state_dict" --max_shard_size 1000GB

# Step 2: Convert pytorch_model.bin to model.safetensors
echo "Step 2: Converting pytorch_model.bin to model.safetensors..."
python src/opentau/scripts/bin_to_safetensors.py "$CHECKPOINT_DIR/full_state_dict/pytorch_model.bin" --output_file "$CHECKPOINT_DIR/model.safetensors"

echo "Conversion completed successfully!"
echo "Model saved as: $CHECKPOINT_DIR/model.safetensors"
