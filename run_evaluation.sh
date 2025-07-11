#!/bin/bash

# Evaluate sampled ImageNet dataset using TissueMNIST pretrained model
# This script runs the evaluation on the sampled dataset

echo "Starting evaluation of sampled ImageNet dataset..."
echo "================================================="

# Set paths
MODEL_PATH="/home/suyoung/Vscode/HAST/models/checkpoints/weights_tissuemnist/resnet18_224_1.pth"
DATA_DIR="/home/suyoung/Vscode/HAST/data/imagenet_balanced_batchnorm"
OUTPUT_FILE="evaluation_results_$(date +%Y%m%d_%H%M%S).txt"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please run the sampling script first to create the dataset."
    exit 1
fi

# Run evaluation
echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

python evaluate_sampled_dataset.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_file "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Evaluation completed successfully!"
    echo "✓ Results saved to: $OUTPUT_FILE"
else
    echo ""
    echo "✗ Evaluation failed!"
    exit 1
fi
