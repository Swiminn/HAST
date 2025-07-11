"""
data loder for loading data
"""
import os
import math
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import struct
import medmnist
from medmnist import INFO

__all__ = ["DataLoader", "PartDataLoader"]


class ImageFolderDataset(data.Dataset):
    """Image Folder Dataset for crawled data"""
    def __init__(self, root_dir, transform=None):
        """
        Initialize Image Folder Dataset
        Args:
            root_dir: Root directory containing images
            transform: Transformation to apply to the images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.image_paths = []
        
        if os.path.isdir(root_dir):
            for filename in os.listdir(root_dir):
                if filename.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(root_dir, filename))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # Return dummy label (0) since we're doing unsupervised learning
            return image, 0
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy tensor if image loading fails
            return torch.zeros(3, 224, 224), 0


class ImageLoader(data.Dataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        class_list = os.listdir(dataset_dir)
        datasets = []
        for cla in class_list:
            cla_path = os.path.join(dataset_dir, cla)
            files = os.listdir(cla_path)
            for file_name in files:
                file_path = os.path.join(cla_path, file_name)
                if os.path.isfile(file_path):
                    # datasets.append((file_path, tuple([float(v) for v in int(cla)])))
                    datasets.append((file_path, [float(cla)]))
                    # print(datasets)
                    # assert False
        
        self.dataset_dir = dataset_dir
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        frames = []
        
        file_path, label = self.datasets[index]
        noise = torch.load(file_path, map_location=torch.device('cpu'))
        return noise, torch.Tensor(label)
    
    def __len__(self):
        return len(self.datasets)


class DataLoader:
    """Data Loader"""
    def __init__(self, train_dataset, test_dataset, batch_size, n_threads=4,
                 ten_crop=False, train_data_path='/home/suyoung/Vscode/HAST/data/', test_data_path='/home/suyoung/Vscode/HAST/data/', logger=None):
        """
        Initialize Data Loader
        Args:
            train_dataset: Training dataset to use
            test_dataset: Test dataset to use
            batch_size: Batch size
            n_threads: Number of threads to use
            ten_crop: Use ten crop for testing
            train_data_path: Path to the training dataset
            test_data_path: Path to the test dataset
            logger: Logger
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.ten_crop = ten_crop
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.logger = logger

        # Determine the type of datasets and load them accordingly
        self.train_loader, self.test_loader = self._load_datasets()

    def _load_datasets(self):
        """Load train and test datasets"""
        # Check if we need to apply grayscale conversion
        convert_to_grayscale = (
            self.test_dataset in ["tissuemnist_28", "tissuemnist_224"] and 
            self.train_dataset in ["imagenet", "crawled_data", "imagenet_balanced"]
        )
        
        if convert_to_grayscale:
            print(f"Converting {self.train_dataset} training data to grayscale for compatibility with {self.test_dataset}")
        
        # Handle training dataset
        if self.train_dataset in ["cifar100", "cifar10"]:
            train_loader = self._load_cifar(self.train_dataset, is_train=True, data_path=self.train_data_path)
        elif self.train_dataset == "imagenet":
            train_loader = self._load_imagenet(is_train=True, data_path=self.train_data_path, convert_to_grayscale=convert_to_grayscale)
        elif self.train_dataset == "imagenet_balanced":
            train_loader = self._load_imagenet_balanced(is_train=True, data_path=self.train_data_path, convert_to_grayscale=convert_to_grayscale)
        elif self.train_dataset in ["dermamnist_28", "dermamnist_224", "tissuemnist_28", "tissuemnist_224"]:
            train_loader = self._load_medmnist(self.train_dataset, is_train=True, data_path=self.train_data_path)
        elif self.train_dataset == "crawled_data":
            train_loader = self._load_crawled_data(data_path=self.train_data_path, convert_to_grayscale=convert_to_grayscale)
        else:
            raise ValueError(f"Unknown training dataset: {self.train_dataset}")
        
        # Handle test dataset
        if self.test_dataset in ["cifar100", "cifar10"]:
            test_loader = self._load_cifar(self.test_dataset, is_train=False, data_path=self.test_data_path)
        elif self.test_dataset == "imagenet":
            test_loader = self._load_imagenet(is_train=False, data_path=self.test_data_path)
        elif self.test_dataset == "imagenet_balanced":
            test_loader = self._load_imagenet_balanced(is_train=False, data_path=self.test_data_path, convert_to_grayscale=convert_to_grayscale)
        elif self.test_dataset in ["dermamnist_28", "dermamnist_224", "tissuemnist_28", "tissuemnist_224"]:
            test_loader = self._load_medmnist(self.test_dataset, is_train=False, data_path=self.test_data_path)
        elif self.test_dataset == "crawled_data":
            test_loader = None  # crawled_data doesn't have test set
        else:
            raise ValueError(f"Unknown test dataset: {self.test_dataset}")
        
        return train_loader, test_loader

    def getloader(self):
        """Get the data loader"""
        return self.train_loader, self.test_loader

    def _load_imagenet(self, is_train=True, data_path=None, convert_to_grayscale=False):
        """Get the ImageNet data loader"""
        if data_path is None:
            data_path = self.train_data_path if is_train else self.test_data_path
        
        if is_train:
            datadir = os.path.join(data_path, "train")
        else:
            datadir = os.path.join(data_path, "val")
            
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if convert_to_grayscale:
            # Convert to grayscale then back to RGB for tissuemnist compatibility
            if is_train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
                    transforms.ToTensor(),
                    normalize])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
                    transforms.ToTensor(),
                    normalize])
        else:
            # Original ImageNet transforms
            if is_train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize])

        dataset = dsets.ImageFolder(datadir, transform)
        
        # Limit ImageNet training dataset to 5000 images
        if is_train:
            train_size = min(5000, len(dataset))
            indices = torch.randperm(len(dataset))[:train_size]
            dataset = torch.utils.data.Subset(dataset, indices)
            print(f"ImageNet training dataset limited to {len(dataset)} images")
        
        # Set n_channels and n_classes from ImageNet (first dataset loaded)
        if not hasattr(self, 'n_channels'):
            self.n_channels = 3  # ImageNet is RGB
            self.n_classes = 1000  # ImageNet has 1000 classes

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=is_train,
            num_workers=self.n_threads,
            pin_memory=True)

        return loader

    def _load_imagenet_balanced(self, is_train=True, data_path=None, convert_to_grayscale=False):
        """Get the ImageNet Balanced data loader (from imagenet_balanced folder)"""
        if data_path is None:
            data_path = self.train_data_path if is_train else self.test_data_path
        
        # imagenet_balanced doesn't have separate train/test splits, so we use the same data
        # and create a split if needed
        datadir = data_path
        
        if not os.path.exists(datadir):
            raise FileNotFoundError(f"ImageNet Balanced dataset not found at {datadir}")
            
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=[.5], std=[.5])

        if convert_to_grayscale:
            # Convert to grayscale then back to RGB for tissuemnist compatibility
            if is_train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
                    transforms.ToTensor(),
                    normalize])
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
                    transforms.ToTensor(),
                    normalize])
        else:
            # Original ImageNet transforms adapted for imagenet_balanced
            if is_train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize])

        # Load dataset using ImageFolder (since it's organized in class folders)
        full_dataset = dsets.ImageFolder(datadir, transform)
        
        # Count images per class
        class_counts = {}
        for _, class_idx in full_dataset.samples:
            class_name = full_dataset.classes[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"ImageNet Balanced dataset loaded from {datadir}")
        print(f"Found {len(full_dataset.classes)} classes with {len(full_dataset)} total images")
        for class_name, count in class_counts.items():
            print(f"  Class {class_name}: {count} images")
        
        # # Create train/test split if needed
        # if is_train:
        #     # Use 80% for training
        #     train_size = int(0.8 * len(full_dataset))
        #     test_size = len(full_dataset) - train_size
        #     train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        #     dataset = train_dataset
        #     print(f"Using {len(dataset)} images for training")
        # else:
        #     # Use 20% for testing
        #     train_size = int(0.8 * len(full_dataset))
        #     test_size = len(full_dataset) - train_size
        #     _, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
        #     dataset = test_dataset
        #     print(f"Using {len(dataset)} images for testing")
        
        # Set n_channels and n_classes from imagenet_balanced
        if not hasattr(self, 'n_channels'):
            self.n_channels = 3  # ImageNet Balanced is RGB
            self.n_classes = len(full_dataset.classes)  # Number of classes in imagenet_balanced

        loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            shuffle=is_train,
            num_workers=self.n_threads,
            pin_memory=True)

        return loader

    def _load_medmnist(self, dataset_name, is_train=True, data_path=None):
        """Get the MedMNIST data loader"""
        if data_path is None:
            data_path = self.train_data_path if is_train else self.test_data_path
        
        # Determine dataset size and info
        if dataset_name == "dermamnist_28":
            data_flag = 'dermamnist'
            size = 28
        elif dataset_name == "dermamnist_224":
            data_flag = 'dermamnist'
            size = 224
        elif dataset_name == "tissuemnist_28":
            data_flag = 'tissuemnist'
            size = 28
        elif dataset_name == "tissuemnist_224":
            data_flag = 'tissuemnist'
            size = 224
        else:
            raise ValueError(f"Unknown MedMNIST dataset: {dataset_name}")
        
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        
        # Set n_channels and n_classes from the first dataset loaded
        if not is_train:
            self.n_channels = info['n_channels']
            self.n_classes = len(info['label'])
        
        # Define transforms
        if size == 28:
            if is_train:
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])
                ])
        else:  # size == 224
            if data_flag == 'tissuemnist':
                # TissueMNIST is 1-channel (grayscale)
                if is_train:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[.5], std=[.5])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[.5], std=[.5])
                    ])
            else:
                # DermaMNIST is 3-channel (RGB)
                if is_train:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[.5], std=[.5])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[.5], std=[.5])
                    ])
        
        # Use absolute path and ensure directory exists
        medmnist_root = os.path.abspath(data_path)
        os.makedirs(medmnist_root, exist_ok=True)
        
        # Load dataset
        split = 'train' if is_train else 'test'
        dataset = DataClass(split=split, transform=transform, download=True, as_rgb=True, root=medmnist_root)
        
        # Limit tissuemnist training dataset to 5000 images
        if data_flag == 'tissuemnist' and is_train:
            train_size = min(5000, len(dataset))
            indices = torch.randperm(len(dataset))[:train_size]
            dataset = torch.utils.data.Subset(dataset, indices)
            print(f"TissueMNIST training dataset limited to {len(dataset)} images")
        
        if is_train:
            print(f"Training dataset: {dataset_name} ({len(dataset)} images)")
        else:
            print(f"Test dataset: {dataset_name} ({len(dataset)} images)")
        
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=is_train,
            num_workers=self.n_threads,
            pin_memory=True
        )
        
        return loader
    
    def _load_cifar(self, dataset_name, is_train=True, data_path=None):
        """Get the CIFAR data loader"""
        if data_path is None:
            data_path = self.train_data_path if is_train else self.test_data_path
            
        if dataset_name == "cifar10":
            norm_mean = [0.49139968, 0.48215827, 0.44653124]
            norm_std = [0.24703233, 0.24348505, 0.26158768]
            dataset_class = dsets.CIFAR10
            num_classes = 10
        elif dataset_name == "cifar100":
            norm_mean = [0.50705882, 0.48666667, 0.44078431]
            norm_std = [0.26745098, 0.25568627, 0.27607843]
            dataset_class = dsets.CIFAR100
            num_classes = 100
        else:
            raise ValueError(f"Invalid cifar dataset: {dataset_name}")

        # Set n_channels and n_classes from the first dataset loaded
        if not hasattr(self, 'n_channels'):
            self.n_channels = 3  # CIFAR is RGB
            self.n_classes = num_classes

        data_root = data_path
        
        # Transform with or without data augmentation
        if is_train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

        dataset = dataset_class(root=data_root,
                               train=is_train,
                               transform=transform,
                               download=True)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=self.batch_size,
                                           shuffle=is_train,
                                           pin_memory=True,
                                           num_workers=self.n_threads)
        
        return loader

    def _load_crawled_data(self, data_path=None, convert_to_grayscale=False):
        """Get the crawled data loader"""
        if data_path is None:
            data_path = self.train_data_path
            
        # For crawled data, we use it for unsupervised learning, so no test loader needed
        if convert_to_grayscale:
            # Convert to grayscale then back to RGB for tissuemnist compatibility
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Original crawled data transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Set n_channels and n_classes from the first dataset loaded
        if not hasattr(self, 'n_channels'):
            self.n_channels = 3  # crawled_data is RGB
            self.n_classes = 1  # dummy class for unsupervised learning
        
        dataset = ImageFolderDataset(data_path, transform)
        
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_threads,
            pin_memory=True
        )
        
        return loader
