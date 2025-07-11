#!/bin/bash
# LION-400M 데이터셋 다운로드 실행 스크립트

echo "=== LION-400M Dataset Downloader ==="
echo "Downloading images only, up to 1TB"
echo "Save location: /hdd1/data/LION-400M"
echo ""

# huggingface_hub 설치 (필요한 경우)
echo "Installing required packages..."
pip install huggingface_hub requests tqdm

echo ""
echo "Starting download..."

# Hugging Face에서 다운로드 시도
python3 download_lion_dataset.py \
    --save_dir "/hdd1/data/LION-400M" \
    --max_size_tb 1.0 \
    --method "urls"

echo ""
echo "Download completed!"
echo "Check /hdd1/data/LION-400M for downloaded images"
