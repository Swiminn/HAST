#!/usr/bin/env python3
"""
Test script for dataloader functionality
"""

import sys
sys.path.append('/home/suyoung/Vscode/HAST')

from dataloader import DataLoader

def test_dataloader():
    """Test different dataset configurations"""
    
    print("=== Testing DataLoader ===\n")
    
    # Test 1: TissueMNIST
    print("1. Testing TissueMNIST 224...")
    try:
        loader = DataLoader(
            train_dataset="tissuemnist_224",
            test_dataset="tissuemnist_224", 
            batch_size=4,
            n_threads=2,
            train_data_path="/home/suyoung/Vscode/HAST/data",
            test_data_path="/home/suyoung/Vscode/HAST/data"
        )
        train_loader, test_loader = loader.getloader()
        print(f"✓ TissueMNIST loaded successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Channels: {loader.n_channels}, Classes: {loader.n_classes}")
    except Exception as e:
        print(f"✗ TissueMNIST failed: {e}")
    
    print()
    
    # Test 2: ImageNet Balanced
    print("2. Testing ImageNet Balanced...")
    try:
        loader = DataLoader(
            train_dataset="imagenet_balanced",
            test_dataset="imagenet_balanced",
            batch_size=4,
            n_threads=2,
            train_data_path="/home/suyoung/Vscode/HAST/data/imagenet_balanced",
            test_data_path="/home/suyoung/Vscode/HAST/data/imagenet_balanced"
        )
        train_loader, test_loader = loader.getloader()
        print(f"✓ ImageNet Balanced loaded successfully")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Channels: {loader.n_channels}, Classes: {loader.n_classes}")
    except Exception as e:
        print(f"✗ ImageNet Balanced failed: {e}")
    
    print()
    
    # Test 3: Test a batch
    print("3. Testing batch loading...")
    try:
        loader = DataLoader(
            train_dataset="imagenet_balanced",
            test_dataset="tissuemnist_224",
            batch_size=2,
            n_threads=1
        )
        train_loader, test_loader = loader.getloader()
        
        # Test train batch
        train_iter = iter(train_loader)
        train_batch = next(train_iter)
        print(f"✓ Train batch shape: {train_batch[0].shape}, {train_batch[1].shape}")
        
        # Test test batch
        test_iter = iter(test_loader)
        test_batch = next(test_iter)
        print(f"✓ Test batch shape: {test_batch[0].shape}, {test_batch[1].shape}")
        
    except Exception as e:
        print(f"✗ Batch loading failed: {e}")

if __name__ == "__main__":
    test_dataloader()
