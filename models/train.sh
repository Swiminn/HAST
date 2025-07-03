#!/bin/bash

# DermaMNIST training script using GPU 1
# This script trains a ResNet18 model from scratch on the DermaMNIST dataset

# Set environment variables
export CUDA_VISIBLE_DEVICES=1

# Dataset and model configuration
DATA_FLAG="dermamnist"
MODEL_FLAG="resnet18"
SIZE=28
BATCH_SIZE=128
OUTPUT_ROOT="./output_train"
RUN="train_dermamnist_resnet18_scratch"

# Training configuration
NUM_EPOCHS=100

echo "Starting DermaMNIST training from scratch..."
echo "Dataset: $DATA_FLAG"
echo "Model: $MODEL_FLAG"
echo "Image Size: $SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Training Epochs: $NUM_EPOCHS"
echo "Training from scratch (no pre-trained weights)"
echo "================================"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_ROOT

# Run training (no model_path specified for training from scratch)
python train_and_eval_pytorch.py \
    --data_flag $DATA_FLAG \
    --model_flag $MODEL_FLAG \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --output_root $OUTPUT_ROOT \
    --run $RUN \
    --download

echo "================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_ROOT"
echo "Training logs and checkpoints available in: $OUTPUT_ROOT/$DATA_FLAG"