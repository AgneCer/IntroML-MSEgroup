import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import os
from PIL import Image

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

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        folder_name = os.path.basename(os.path.dirname(path))
        target = folder_name.split('_')[0]
        
        return sample, target

def get_data_loaders(data_dir, batch_size_train, batch_size_test, num_workers, train_transform=None, val_transform=None):
    
    if not train_transform:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if not val_transform:
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load the dataset from the directory
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    # Calculate train/validation split sizes
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Apply the appropriate transforms to each subset
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
