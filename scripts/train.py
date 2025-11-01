"""
GPU-Accelerated CNN Training on CIFAR-10
Implements mixed precision training and parallel data loading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class CIFAR10CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification
    Architecture optimized for GPU acceleration
    """
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Block 1
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Block 2
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Block 3
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Block 4
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 512 * 2 * 2)
        
        # Fully connected layers with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def get_data_loaders(batch_size=128, num_workers=4):
    """
    Create parallelized data loaders with augmentation
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                     download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                    download=True, transform=test_transform)
    
    # Parallel data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, 
                             pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, 
                            pin_memory=True, persistent_workers=True)
    
    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """
    Train for one epoch with mixed precision
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training with autocast
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Scaled backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'[] Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

def main():
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Version: {torch.version.cuda}')
    
    # Hyperparameters
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.001
    
    # Data loaders
    print('[] Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    
    # Model
    print('[] Initializing CNN model...')
    model = CIFAR10CNN(num_classes=10).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    print('[] Starting training...')
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f'\n[] Epoch {epoch+1}/{num_epochs}')
        print(f'[] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'[] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print(f'[] Epoch Time: {epoch_time:.2f}s')
        print(f'[] Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, 'best_model.pth')
            print(f'[] Saved best model with accuracy: {best_acc:.2f}%')
    
    total_time = time.time() - start_time
    print(f'\n[] Training completed in {total_time/60:.2f} minutes')
    print(f'[] Best Test Accuracy: {best_acc:.2f}%')
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print('[] Final model saved')

if __name__ == '__main__':
    main()
