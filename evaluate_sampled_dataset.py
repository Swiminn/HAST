"""
Evaluate the sampled ImageNet dataset using the same TissueMNIST pretrained model
Measure how many samples are predicted for each class
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import argparse
from torchvision.models import resnet18
import medmnist
from medmnist import INFO
from collections import defaultdict
import numpy as np


class NewBatchNorm2d(nn.Module):
    """
    Custom BatchNorm layer that uses current input statistics for normalization
    but applies frozen beta and gamma parameters from pretrained model
    """
    def __init__(self, original_bn):
        super(NewBatchNorm2d, self).__init__()
        self.num_features = original_bn.num_features
        self.eps = original_bn.eps
        self.momentum = original_bn.momentum
        
        # Store original parameters (frozen gamma, beta)
        self.register_buffer('frozen_weight', original_bn.weight.clone().detach())
        self.register_buffer('frozen_bias', original_bn.bias.clone().detach())
        
        # These will be computed from current batch
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_var', torch.ones(self.num_features))
        
        # Store reference to original BN for debugging
        self.original_bn = original_bn
    
    def forward(self, input):
        # Always use current batch statistics for mean and variance
        if self.training:
            # Calculate mean and variance from current batch
            batch_mean = input.mean([0, 2, 3], keepdim=False)
            batch_var = input.var([0, 2, 3], unbiased=False, keepdim=False)
            
            # # Update running statistics (for tracking purposes)
            # if self.momentum is None:
            #     exponential_average_factor = 0.0
            # else:
            #     exponential_average_factor = self.momentum
            
            # with torch.no_grad():
            #     self.running_mean.mul_(1 - exponential_average_factor).add_(batch_mean, alpha=exponential_average_factor)
            #     self.running_var.mul_(1 - exponential_average_factor).add_(batch_var, alpha=exponential_average_factor)
            
            # Use batch statistics for normalization
            mean = batch_mean.detach()
            var = batch_var.detach()
        else:
            # Even in eval mode, use current batch statistics
            mean = input.mean([0, 2, 3], keepdim=False).detach()
            var = input.var([0, 2, 3], unbiased=False, keepdim=False).detach()
        
        # Apply batch normalization with frozen gamma/beta
        return F.batch_norm(
            input, 
            mean.detach(), 
            var.detach(), 
            self.frozen_weight,  # Use frozen pretrained gamma
            self.frozen_bias,    # Use frozen pretrained beta
            training=False,      # We handle statistics ourselves
            momentum=0.0,        # No momentum since we handle it ourselves
            eps=self.eps
        )


class SampledDataset(Dataset):
    """Dataset class for the sampled ImageNet images"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_names = []
        
        # Load all images from class directories
        for class_idx in range(8):  # 8 classes for TissueMNIST
            class_dir = os.path.join(data_dir, f"class_{class_idx}")
            if os.path.exists(class_dir):
                self.class_names.append(f"class_{class_idx}")
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(class_dir, filename)
                        self.samples.append((image_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images from {len(self.class_names)} classes")
        
        # Print per-class counts
        class_counts = defaultdict(int)
        for _, class_idx in self.samples:
            class_counts[class_idx] += 1
        
        print("Per-class image counts:")
        for class_idx in range(8):
            count = class_counts[class_idx]
            print(f"  Class {class_idx}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, true_class = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image in case of error
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, true_class, image_path


class ModelEvaluator:
    def __init__(self, model_path, data_dir):
        """
        Initialize the model evaluator
        Args:
            model_path: Path to the TissueMNIST pretrained model
            data_dir: Path to the sampled dataset directory
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 8
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Setup dataset and dataloader
        self.dataset, self.dataloader = self._setup_dataset()
        
        # Results storage
        self.results = {
            'true_classes': [],
            'predicted_classes': [],
            'probabilities': [],
            'image_paths': []
        }
    
    def _replace_batchnorm_layers(self, model):
        """
        Replace all BatchNorm layers with NewBatchNorm2d layers
        that use current batch statistics but frozen gamma/beta
        """
        print("Replacing BatchNorm layers with custom BatchNorm (batch stats + frozen gamma/beta)...")
        
        def replace_bn_recursive(module, name=""):
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    # Create new BN layer with frozen parameters
                    new_bn = NewBatchNorm2d(child_module)
                    new_bn = new_bn.to(child_module.weight.device)
                    
                    # Replace the module
                    setattr(module, child_name, new_bn)
                    print(f"  Replaced {full_name} with NewBatchNorm2d")
                else:
                    # Recursively replace in child modules
                    replace_bn_recursive(child_module, full_name)
        
        replace_bn_recursive(model)
        print("✓ BatchNorm replacement completed")
        return model
    
    def _load_model(self, model_path):
        """Load the TissueMNIST pretrained model and replace BatchNorm layers"""
        print("Loading TissueMNIST pretrained model...")
        
        # Get number of classes from TissueMNIST info
        info = INFO['tissuemnist']
        n_classes = len(info['label'])
        
        # Create ResNet18 model
        model = resnet18(weights=None, num_classes=n_classes)
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'net' in checkpoint:
                    model.load_state_dict(checkpoint['net'], strict=True)
                else:
                    model.load_state_dict(checkpoint, strict=True)
                print(f"✓ Loaded TissueMNIST pretrained model from {model_path}")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                raise e
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Replace BatchNorm layers with custom implementation
        model = self._replace_batchnorm_layers(model)
        
        model.to(self.device)
        model.train()  # Set to training mode so NewBatchNorm2d uses current batch statistics
        return model
    
    def _setup_dataset(self):
        """Setup the sampled dataset"""
        print("Setting up sampled dataset...")
        
        # Transform for images (same as used in sampling)
        transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])])
        
        dataset = SampledDataset(self.data_dir, transform=transform)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return dataset, dataloader
    
    def evaluate(self):
        """Evaluate the model on the sampled dataset"""
        print("\nEvaluating model on sampled dataset...")
        print("Note: Using custom BatchNorm with current batch statistics and frozen gamma/beta")
        
        # Initialize counters
        true_class_counts = [0] * self.num_classes
        predicted_class_counts = [0] * self.num_classes
        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        total_samples = 0
        correct_predictions = 0
        
        # Progress bar
        pbar = tqdm(total=len(self.dataset), desc="Evaluating")
        
        with torch.no_grad():
            for batch_idx, (images, true_classes, image_paths) in enumerate(self.dataloader):
                images = images.to(self.device)
                true_classes = true_classes.to(self.device)
                
                # Get model predictions
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted_classes = torch.max(outputs, 1)
                
                # Process batch results
                for i in range(images.size(0)):
                    true_class = true_classes[i].item()
                    predicted_class = predicted_classes[i].item()
                    probs = probabilities[i].cpu().numpy()
                    
                    # Store results
                    self.results['true_classes'].append(true_class)
                    self.results['predicted_classes'].append(predicted_class)
                    self.results['probabilities'].append(probs)
                    self.results['image_paths'].append(image_paths[i])
                    
                    # Update counters
                    true_class_counts[true_class] += 1
                    predicted_class_counts[predicted_class] += 1
                    confusion_matrix[true_class][predicted_class] += 1
                    
                    if true_class == predicted_class:
                        correct_predictions += 1
                    
                    total_samples += 1
                    pbar.update(1)
        
        pbar.close()
        
        # Calculate accuracy
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        # Print results
        self._print_results(true_class_counts, predicted_class_counts, confusion_matrix, accuracy, total_samples)
        
        return self.results
    
    def _print_results(self, true_class_counts, predicted_class_counts, confusion_matrix, accuracy, total_samples):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("=== Evaluation Results ===")
        print("="*80)
        
        # Calculate correct predictions from accuracy
        correct_predictions = int(accuracy * total_samples)
        
        print(f"Total samples evaluated: {total_samples}")
        print(f"Overall accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
        
        print("\n=== True Class Distribution (from sampled dataset) ===")
        for class_idx in range(self.num_classes):
            count = true_class_counts[class_idx]
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"Class {class_idx}: {count:4d} images ({percentage:5.1f}%)")
        
        print("\n=== Predicted Class Distribution (model predictions) ===")
        for class_idx in range(self.num_classes):
            count = predicted_class_counts[class_idx]
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"Class {class_idx}: {count:4d} images ({percentage:5.1f}%)")
        
        print("\n=== Per-Class Accuracy ===")
        for class_idx in range(self.num_classes):
            correct = confusion_matrix[class_idx][class_idx]
            total = true_class_counts[class_idx]
            acc = (correct / total * 100) if total > 0 else 0
            print(f"Class {class_idx}: {correct:3d}/{total:3d} correct ({acc:5.1f}%)")
        
        print("\n=== Confusion Matrix ===")
        print("True\\Pred", end="")
        for j in range(self.num_classes):
            print(f"{j:6d}", end="")
        print()
        
        for i in range(self.num_classes):
            print(f"Class {i:2d}", end="")
            for j in range(self.num_classes):
                print(f"{confusion_matrix[i][j]:6d}", end="")
            print()
        
        # Check class distribution alignment
        print("\n=== Class Distribution Comparison ===")
        print("Class | True Count | Pred Count | Ratio (Pred/True)")
        print("-" * 50)
        for class_idx in range(self.num_classes):
            true_count = true_class_counts[class_idx]
            pred_count = predicted_class_counts[class_idx]
            ratio = (pred_count / true_count) if true_count > 0 else float('inf')
            print(f"{class_idx:5d} | {true_count:10d} | {pred_count:10d} | {ratio:12.2f}")
    
    def save_detailed_results(self, output_file="evaluation_results.txt"):
        """Save detailed results to file"""
        with open(output_file, 'w') as f:
            f.write("Detailed Evaluation Results\n")
            f.write("="*50 + "\n\n")
            
            for i, (true_class, pred_class, probs, img_path) in enumerate(zip(
                self.results['true_classes'],
                self.results['predicted_classes'], 
                self.results['probabilities'],
                self.results['image_paths']
            )):
                f.write(f"Sample {i+1:5d}: {os.path.basename(img_path)}\n")
                f.write(f"  True class: {true_class}, Predicted: {pred_class}\n")
                f.write(f"  Probabilities: {probs}\n")
                f.write(f"  Max prob: {np.max(probs):.4f} for class {np.argmax(probs)}\n")
                f.write(f"  Correct: {'Yes' if true_class == pred_class else 'No'}\n\n")
        
        print(f"Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate sampled ImageNet dataset using TissueMNIST pretrained model')
    parser.add_argument('--model_path', type=str, 
                        default="/home/suyoung/Vscode/HAST/models/checkpoints/weights_tissuemnist/resnet18_224_1.pth",
                        help='Path to TissueMNIST pretrained model')
    parser.add_argument('--data_dir', type=str, 
                        default="/home/suyoung/Vscode/HAST/data/imagenet_balanced_batchnorm",
                        help='Path to sampled dataset directory')
    parser.add_argument('--output_file', type=str, 
                        default="evaluation_results.txt",
                        help='Output file for detailed results')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please make sure the sampled dataset exists.")
        return
    
    # Create evaluator and run evaluation
    try:
        evaluator = ModelEvaluator(args.model_path, args.data_dir)
        results = evaluator.evaluate()
        evaluator.save_detailed_results(args.output_file)
        
        print(f"\nEvaluation completed!")
        print(f"Detailed results saved to: {args.output_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
