#!/usr/bin/env python3
"""
Test script for trainer modifications
"""

import sys
import os
sys.path.append('/home/suyoung/Vscode/HAST')

import torch
import torch.nn as nn
from dataloader import DataLoader
from models.models import resnet18
from trainer_direct import Trainer
import utils
from utils import log_print

# Simple mock settings class
class MockSettings:
    def __init__(self):
        self.nEpochs = 2
        self.momentum = 0.9
        self.weightDecay = 1e-4
        self.alpha = 0.5
        self.temperature = 4.0
        self.lam = 0.1
        self.train_dataset = "imagenet_balanced"
        self.test_dataset = "imagenet_balanced"
        self.nClasses = 8

# Simple mock args class
class MockArgs:
    def __init__(self):
        self.local_rank = 0

def test_trainer_modifications():
    """Test the trainer modifications"""
    print("=== Testing Trainer Modifications ===\n")
    
    try:
        # Create data loaders
        print("1. Creating data loaders...")
        data_loader = DataLoader(
            train_dataset="imagenet_balanced",
            test_dataset="imagenet_balanced",
            batch_size=4,
            n_threads=1,
            train_data_path="/home/suyoung/Vscode/HAST/data/imagenet_balanced",
            test_data_path="/home/suyoung/Vscode/HAST/data/imagenet_balanced"
        )
        train_loader, test_loader = data_loader.getloader()
        print(f"✓ Data loaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        # Create models
        print("\n2. Creating models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Student model
        student_model = resnet18(weights=None, num_classes=8)
        student_model = nn.DataParallel(student_model).to(device)
        
        # Teacher model (same architecture for simplicity)
        teacher_model = resnet18(weights=None, num_classes=8)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        
        print(f"✓ Models created on device: {device}")
        
        # Create learning rate schedulers
        print("\n3. Creating learning rate schedulers...")
        lr_master_S = utils.LRPolicy(0.001, 'multi_step', [0.5, 0.75], 0.1)
        lr_master_G = utils.LRPolicy(0.001, 'multi_step', [0.5, 0.75], 0.1)
        print("✓ Learning rate schedulers created")
        
        # Create settings and args
        settings = MockSettings()
        args = MockArgs()
        
        # Create logger
        logger = log_print.Logger("test_log.txt")
        
        # Create trainer
        print("\n4. Creating trainer...")
        trainer = Trainer(
            model=student_model,
            model_teacher=teacher_model,
            lr_master_S=lr_master_S,
            lr_master_G=lr_master_G,
            train_loader=train_loader,
            test_loader=test_loader,
            settings=settings,
            args=args,
            logger=logger
        )
        print("✓ Trainer created successfully")
        
        # Test one epoch of training (just a few batches)
        print("\n5. Testing training with class counting...")
        print("Running a few training batches to test class counting...")
        
        # Temporarily limit the train loader to just a few batches for testing
        limited_train_data = []
        for i, batch in enumerate(train_loader):
            limited_train_data.append(batch)
            if i >= 2:  # Just 3 batches for testing
                break
        
        # Mock train loader with limited data
        class MockTrainLoader:
            def __init__(self, data):
                self.data = data
            def __iter__(self):
                return iter(self.data)
            def __len__(self):
                return len(self.data)
        
        trainer.train_loader = MockTrainLoader(limited_train_data)
        
        # Run one epoch
        top1_error, top1_loss, top5_error = trainer.train(epoch=0)
        
        print(f"✓ Training completed successfully!")
        print(f"  Top1 Error: {top1_error:.4f}")
        print(f"  Top1 Loss: {top1_loss:.4f}")
        print(f"  Top5 Error: {top5_error:.4f}")
        
        print("\n=== Test Completed Successfully ===")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trainer_modifications()
