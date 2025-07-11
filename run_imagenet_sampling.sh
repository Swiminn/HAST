#!/bin/bash
# ImageNet에서 이미지 샘플링 실행 스크립트

echo "Starting ImageNet sampling with TissueMNIST pretrained model..."

python3 sample_image_from_imagenet.py \
    --model_path "/home/suyoung/Vscode/HAST/models/checkpoints/weights_tissuemnist/resnet18_224_1.pth" \
    --imagenet_root "/home/suyoung/Vscode/HAST/data/imagenet" \
    --save_dir "/home/suyoung/Vscode/HAST/data/imagenet_balanced_batchnorm" \
    --batch_size 64 \
    --max_samples 1281167

echo "Sampling completed!"
