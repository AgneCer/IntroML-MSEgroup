import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(batch_size_train, batch_size_test, num_workers):
    # Define the data transformations for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the data transformations for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download the Flowers102 dataset
    train_dataset = datasets.Flowers102(root='data', split='train', download=True, transform=train_transform)
    val_dataset = datasets.Flowers102(root='data', split='val', download=True, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
