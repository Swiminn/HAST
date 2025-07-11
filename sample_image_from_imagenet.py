"""
Sample images from ImageNet dataset using TissueMNIST pretrained model
Sample 5000 images (625 images per class for 8 classes) based on model predictions
For each class, select the 625 images with the highest predicted probability for that class
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from PIL import Image
import random
import shutil
from tqdm import tqdm
import argparse
from torchvision.models import resnet18
import medmnist
from medmnist import INFO
import heapq
from collections import defaultdict


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
            mean, 
            var, 
            self.frozen_weight,  # Use frozen pretrained gamma
            self.frozen_bias,    # Use frozen pretrained beta
            training=False,      # We handle statistics ourselves
            momentum=0.0,        # No momentum since we handle it ourselves
            eps=self.eps
        )

class ImageNetSampler:
    def __init__(self, model_path, imagenet_root, save_dir="/home/suyoung/Vscode/HAST/data/imagenet_balanced"):
        """
        Initialize the ImageNet sampler
        Args:
            model_path: Path to the TissueMNIST pretrained model
            imagenet_root: Path to ImageNet dataset root directory
            save_dir: Directory to save sampled images
        """
        self.imagenet_root = imagenet_root
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Target counts per class (8 classes, 625 images each = 5000 total)
        self.target_per_class = 625
        self.num_classes = 8
        
        # Use min-heaps to track top N images per class
        # Each heap stores tuples of (probability, image_path, image_index)
        self.class_heaps = [[] for _ in range(self.num_classes)]
        
        # Create save directories
        os.makedirs(self.save_dir, exist_ok=True)
        for class_idx in range(self.num_classes):
            class_dir = os.path.join(self.save_dir, f"class_{class_idx}")
            os.makedirs(class_dir, exist_ok=True)
        
        # Load the TissueMNIST pretrained model
        self.model = self._load_model(model_path)
        
        # Image preprocessing to match TissueMNIST training
        self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])])
        
        # Setup ImageNet dataset
        self.dataset = self._setup_dataset()
    
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
        model.eval()  # Set to evaluation mode
        return model
    
    def _setup_dataset(self):
        """Setup ImageNet dataset"""
        print("Setting up ImageNet dataset...")
        
        # Transform for ImageNet images
        imagenet_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            # Load ImageNet training set
            dataset = ImageNet(
                root=self.imagenet_root,
                split='train',
                transform=self.transform
            )
            print(f"✓ ImageNet dataset loaded: {len(dataset)} images")
            
            # Create a custom dataset that returns both image and path
            class ImageNetWithPaths(torch.utils.data.Dataset):
                def __init__(self, imagenet_dataset):
                    self.imagenet_dataset = imagenet_dataset
                    self.samples = imagenet_dataset.samples
                
                def __len__(self):
                    return len(self.imagenet_dataset)
                
                def __getitem__(self, idx):
                    image, label = self.imagenet_dataset[idx]
                    image_path, _ = self.samples[idx]
                    return image, label, image_path
            
            return ImageNetWithPaths(dataset)
            
        except Exception as e:
            print(f"✗ Error loading ImageNet dataset: {e}")
            print(f"Make sure ImageNet is properly downloaded at: {self.imagenet_root}")
            raise e
    
    def _predict_batch(self, image_batch):
        """Predict classes and probabilities for a batch of images"""
        try:
            image_batch = image_batch.to(self.device)
            
            with torch.no_grad():
                output = self.model(image_batch)
                probabilities = F.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                # Convert to list of prediction results
                batch_results = []
                for i in range(image_batch.size(0)):
                    predicted_class = predicted[i].item()
                    class_probability = probabilities[i, predicted_class].item()
                    all_probs = probabilities[i].cpu().numpy()
                    batch_results.append((predicted_class, class_probability, all_probs))
                
                return batch_results
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            # Return default values for all images in batch
            batch_size = image_batch.size(0) if hasattr(image_batch, 'size') else len(image_batch)
            return [(-1, 0.0, None) for _ in range(batch_size)]

    def _predict_class_with_probs(self, image_tensor):
        """Predict class and get probabilities using TissueMNIST pretrained model (single image)"""
        try:
            # Add batch dimension if not present
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                # Return predicted class and the probability for that class
                predicted_class = predicted.item()
                class_probability = probabilities[0, predicted_class].item()
                
                return predicted_class, class_probability, probabilities[0].cpu().numpy()
        except Exception as e:
            print(f"Error in prediction: {e}")
            return -1, 0.0, None

    def _add_to_heap(self, class_idx, probability, image_path, image_index):
        """Add image to the appropriate class heap, maintaining top-N images"""
        heap = self.class_heaps[class_idx]
        
        if len(heap) < self.target_per_class:
            # Heap not full, just add
            heapq.heappush(heap, (probability, image_path, image_index))
        elif probability > heap[0][0]:  # heap[0] is the minimum
            # Replace the lowest probability image
            heapq.heapreplace(heap, (probability, image_path, image_index))

    def _save_selected_images(self):
        """Save the selected top images for each class"""
        print("\nSaving selected images...")
        total_saved = 0
        
        for class_idx in range(self.num_classes):
            heap = self.class_heaps[class_idx]
            class_dir = os.path.join(self.save_dir, f"class_{class_idx}")
            
            # Clear existing images in class directory
            for existing_file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, existing_file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Sort heap by probability (descending order)
            sorted_images = sorted(heap, key=lambda x: x[0], reverse=True)
            
            saved_count = 0
            for i, (probability, image_path, _) in enumerate(sorted_images):
                filename = f"image_{i:04d}_prob_{probability:.4f}.jpg"
                save_path = os.path.join(class_dir, filename)
                
                try:
                    # Copy the original image to the new location
                    shutil.copy2(image_path, save_path)
                    saved_count += 1
                except Exception as e:
                    print(f"Error saving image {image_path}: {e}")
            
            total_saved += saved_count
            print(f"Class {class_idx}: saved {saved_count} images (avg prob: {sum(x[0] for x in sorted_images)/len(sorted_images):.4f})")
        
        return total_saved
    
    def sample_images(self, batch_size=64, max_samples=50000):
        """Sample images from ImageNet using the pretrained model"""
        print("Starting image sampling from ImageNet...")
        print(f"Target: {self.target_per_class * self.num_classes} total images ({self.target_per_class} per class)")
        print("Collecting images based on MODEL PREDICTIONS (highest probability per predicted class)...")
        print("Note: Using custom BatchNorm with current batch statistics and frozen gamma/beta")
        print("Strategy: Each image goes to its PREDICTED class folder, then select top 625 by probability")
        
        # Set model to training mode so NewBatchNorm2d uses current batch statistics
        self.model.train()
        
        # Create data loader with shuffling
        dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        processed_samples = 0
        candidate_counts = [0] * self.num_classes  # Track candidates found per class
        
        # Progress bar
        pbar = tqdm(total=max_samples, desc="Processing images")
        
        for batch_idx, (images, labels, image_paths) in enumerate(dataloader):
            if processed_samples >= max_samples:
                break
            
            # Process batch through model to get predictions
            batch_predictions = self._predict_batch(images)
            
            # Process each image in the batch
            for i, (image_tensor, label, image_path) in enumerate(zip(images, labels, image_paths)):
                if processed_samples >= max_samples:
                    break
                
                processed_samples += 1
                
                # Get prediction results for this image
                predicted_class, class_probability, all_probs = batch_predictions[i]
                
                # Only add to the predicted class heap (not all classes)
                # This ensures consistency with evaluation logic
                if predicted_class >= 0 and predicted_class < self.num_classes:
                    # Add to heap for the predicted class only
                    self._add_to_heap(predicted_class, class_probability, image_path, processed_samples)
                    if class_probability > 0.1:  # Count as significant candidate
                        candidate_counts[predicted_class] += 1
                
                # Update progress bar
                pbar.update(1)
                if processed_samples % 1000 == 0:
                    # Show current heap sizes and minimum probabilities
                    heap_info = []
                    for class_idx in range(self.num_classes):
                        heap_size = len(self.class_heaps[class_idx])
                        min_prob = min(self.class_heaps[class_idx], key=lambda x: x[0])[0] if heap_size > 0 else 0.0
                        heap_info.append(f"C{class_idx}:{heap_size}({min_prob:.3f})")
                    
                    pbar.set_postfix({
                        'processed': processed_samples,
                        'heaps': ' '.join(heap_info),
                        'candidates': f"{sum(candidate_counts)}"
                    })
        
        pbar.close()
        
        # Print collection summary
        print(f"\nProcessed {processed_samples} images")
        for class_idx in range(self.num_classes):
            heap_size = len(self.class_heaps[class_idx])
            candidates = candidate_counts[class_idx]
            min_prob = min(self.class_heaps[class_idx], key=lambda x: x[0])[0] if heap_size > 0 else 0.0
            max_prob = max(self.class_heaps[class_idx], key=lambda x: x[0])[0] if heap_size > 0 else 0.0
            print(f"Class {class_idx}: {heap_size} selected images (candidates: {candidates}, prob range: {min_prob:.4f}-{max_prob:.4f})")
        
        # Save the selected images
        total_saved = self._save_selected_images()
        
        print(f"\nSampling completed! Total images saved: {total_saved}")
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final sampling summary"""
        print("\n" + "="*60)
        print("=== Sampling Summary ===")
        print("="*60)
        
        total_saved = 0
        for class_idx in range(self.num_classes):
            heap_size = len(self.class_heaps[class_idx])
            total_saved += heap_size
        
        print(f"Total images saved: {total_saved}/{self.target_per_class * self.num_classes}")
        print(f"Images saved to: {self.save_dir}")
        print("\nPer-class breakdown:")
        
        for class_idx in range(self.num_classes):
            heap_size = len(self.class_heaps[class_idx])
            percentage = (heap_size / self.target_per_class * 100) if self.target_per_class > 0 else 0
            
            if heap_size > 0:
                sorted_heap = sorted(self.class_heaps[class_idx], key=lambda x: x[0], reverse=True)
                min_prob = sorted_heap[-1][0]  # lowest probability
                max_prob = sorted_heap[0][0]   # highest probability
                avg_prob = sum(x[0] for x in sorted_heap) / len(sorted_heap)
                print(f"  Class {class_idx}: {heap_size:3d}/{self.target_per_class} images ({percentage:5.1f}%) | "
                      f"prob range: {min_prob:.4f}-{max_prob:.4f} (avg: {avg_prob:.4f})")
            else:
                print(f"  Class {class_idx}: {heap_size:3d}/{self.target_per_class} images ({percentage:5.1f}%) | no images found")
        
        # Check if any class is incomplete
        incomplete_classes = [i for i in range(self.num_classes) if len(self.class_heaps[i]) < self.target_per_class]
        if incomplete_classes:
            print(f"\nNote: Classes {incomplete_classes} did not reach target count.")
            print("This might indicate that the ImageNet dataset doesn't contain enough images")
            print("that the TissueMNIST model assigns high probabilities to for these categories.")
        else:
            print(f"\n✓ All classes reached target count of {self.target_per_class} images!")
            print("✓ Selected images are those with highest probabilities for each class.")


def main():
    parser = argparse.ArgumentParser(description='Sample images from ImageNet using TissueMNIST pretrained model')
    parser.add_argument('--model_path', type=str, 
                        default="/home/suyoung/Vscode/HAST/models/checkpoints/weights_tissuemnist/resnet18_224_1.pth",
                        help='Path to TissueMNIST pretrained model')
    parser.add_argument('--imagenet_root', type=str, 
                        default="/home/suyoung/Vscode/HAST/data/imagenet",
                        help='Path to ImageNet dataset root directory')
    parser.add_argument('--save_dir', type=str, 
                        default='/home/suyoung/Vscode/HAST/data/imagenet_balanced',
                        help='Directory to save sampled images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--max_samples', type=int, default=100000,
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Check if ImageNet dataset exists
    if not os.path.exists(args.imagenet_root):
        print(f"Error: ImageNet dataset not found: {args.imagenet_root}")
        print("Please make sure ImageNet dataset is downloaded and extracted.")
        return
    
    # Create sampler and start sampling
    try:
        sampler = ImageNetSampler(args.model_path, args.imagenet_root, args.save_dir)
        sampler.sample_images(batch_size=args.batch_size, max_samples=args.max_samples)
    except Exception as e:
        print(f"Error during sampling: {e}")
        return


if __name__ == "__main__":
    main()
