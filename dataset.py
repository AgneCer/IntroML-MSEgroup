import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import numpy as np
import json


def get_data_flowers(batch_size_train, batch_size_test, num_workers, transform=None):

    if not transform:
        # Define the data transformations for training
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define the data transformations for validation
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transform
        val_transform = transform

    # Download the Flowers102 dataset
    train_dataset = datasets.Flowers102(root='data', split='train', download=True, transform=train_transform)
    val_dataset = datasets.Flowers102(root='data', split='val', download=True, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader



def calculate_mean_std(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images.size(0)

    mean /= total_images_count
    std /= total_images_count

    return mean, std

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.labels = []

        # Create a list of image paths and their corresponding labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class CustomTestImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Create a list of image paths
        for img_name in os.listdir(root_dir):
            img_path = os.path.join(root_dir, img_name)
            self.image_paths.append((img_path, os.path.splitext(img_name)[0]))  # Store both image path and ID
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path, img_id = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, img_id

def create_dataloader(root_dir, batch_size=32, img_size=224, val_split=0.2, mode='train', mean=None, std=None, stats_dir='stats', transform=None):
    if mode == 'train':
        if transform is None:
            # Step 1: Use ImageFolder to load the dataset for mean and std calculation
            temp_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
            temp_dataset = datasets.ImageFolder(root=root_dir, transform=temp_transform)

            # Calculate mean and standard deviation
            mean, std = calculate_mean_std(temp_dataset)
            print("Calculated Mean:", mean)
            print("Calculated Std:", std)

            # Save mean and std to a file
            os.makedirs(stats_dir, exist_ok=True)
            with open(os.path.join(stats_dir, 'mean_std.json'), 'w') as f:
                json.dump({'mean': mean.tolist(), 'std': std.tolist()}, f)

            # Define the training and validation transformations
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean.tolist(), std=std.tolist())
            ])

            val_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean.tolist(), std=std.tolist())
            ])
        else:
            train_transform = transform
            val_transform = transform

        # Create an instance of the custom dataset with the training transformations
        full_dataset = CustomImageDataset(root_dir=root_dir, transform=train_transform)

        # Split the dataset into training and validation sets
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Apply the validation transformation to the validation dataset
        val_dataset.dataset.transform = val_transform

        # Create DataLoaders for training and validation sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, mean, std

    elif mode == 'test':
        # Define the test transformations
        if transform is None:
            # Load mean and std from file
            with open(os.path.join(stats_dir, 'mean_std.json'), 'r') as f:
                stats = json.load(f)
                mean = torch.tensor(stats['mean'])
                std = torch.tensor(stats['std'])

            test_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            test_transform = transform

        # Create an instance of the custom test dataset with the test transformations
        test_dataset = CustomTestImageDataset(root_dir=root_dir, transform=test_transform)

        # Create a DataLoader for the test dataset
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return test_loader
