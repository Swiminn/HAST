#!/bin/bash

# DermaMNIST evaluation script using GPU 1
# This script evaluates a pre-trained ResNet18 model on the DermaMNIST dataset

# Set environment variables
export CUDA_VISIBLE_DEVICES=1

# Dataset and model configuration
DATA_FLAG="tissuemnist"
MODEL_FLAG="resnet18"
MODEL_PATH="/home/suyoung/Vscode/SynQ/src/models/checkpoints/weights_tissuemnist/resnet18_28_1.pth"
GPU_IDS="1"
SIZE=28
BATCH_SIZE=128
OUTPUT_ROOT="./output_eval"
RUN="eval_dermamnist_resnet18"

# Set number of epochs to 0 for evaluation only
NUM_EPOCHS=0

echo "Starting TissueMNIST evaluation..."
echo "Dataset: $DATA_FLAG"
echo "Model: $MODEL_FLAG"
echo "Model Path: $MODEL_PATH"
echo "Image Size: $SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Evaluation Only (num_epochs=0)"
echo "================================"

# Run evaluation
python train_and_eval_pytorch.py \
    --data_flag $DATA_FLAG \
    --model_flag $MODEL_FLAG \
    --model_path $MODEL_PATH \
    --size $SIZE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --output_root $OUTPUT_ROOT \
    --run $RUN \
    --download

echo "================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_ROOT"