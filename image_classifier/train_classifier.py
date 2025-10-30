"""
Hornet Image Classifier Training Script
Uses ResNet50 with transfer learning for 3-class classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
import time
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class HornetDataset(Dataset):
    """
    Dataset class for hornet images.
    Expects folder structure: data/train/images/[class_name]/image.jpg
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        # Class names from folders
        self.class_names = ['Vespa_crabro', 'Vespa_velutina', 'Vespula_sp']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Find all images
        self.images = []
        self.labels = []

        images_dir = self.root_dir / split / 'images'

        for class_name in self.class_names:
            class_dir = images_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

        print(f"\nLoaded {split} dataset: {len(self.images)} images")
        for class_name in self.class_names:
            count = self.labels.count(self.class_to_idx[class_name])
            print(f"  {class_name}: {count} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def create_model(num_classes=3, pretrained=True):
    """
    Create ResNet50 model with transfer learning.
    """
    model = models.resnet50(pretrained=pretrained)

    # Freeze early layers (transfer learning)
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for our 3 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc, np.array(all_preds), np.array(all_labels)


def main():
    print("=" * 70)
    print("HORNET IMAGE CLASSIFIER - TRAINING")
    print("=" * 70)

    # Configuration
    DATA_ROOT = Path(r"C:\Users\Zdravkovic\Downloads\archive\data3000\data")
    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_CLASSES = 3

    device = torch.device('cpu')  # Using CPU (GPU would be faster)
    print(f"\nDevice: {device}")

    # Data transforms
    print("\nSetting up data transforms...")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = HornetDataset(DATA_ROOT, split='train', transform=train_transform)
    val_dataset = HornetDataset(DATA_ROOT, split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    print("\nCreating ResNet50 model...")
    print("Downloading pre-trained weights (this may take a moment)...")
    model = create_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    best_val_acc = 0.0
    training_history = []

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

        # Learning rate step
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'time': float(epoch_time)
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.class_names
            }, MODEL_DIR / 'best_model.pth')
            print(f"  [NEW BEST MODEL SAVED]")
        print()

    total_time = time.time() - start_time

    # Final evaluation
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model and generate detailed report
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(MODEL_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device)

    # Classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(final_labels, final_preds,
                                target_names=train_dataset.class_names))

    # Confusion matrix
    print("CONFUSION MATRIX")
    print("=" * 70)
    cm = confusion_matrix(final_labels, final_preds)
    print("Rows = Actual, Columns = Predicted")
    print(f"\n{'':20} {train_dataset.class_names[0]:15} {train_dataset.class_names[1]:15} {train_dataset.class_names[2]:15}")
    for i, class_name in enumerate(train_dataset.class_names):
        print(f"{class_name:20} {cm[i,0]:15} {cm[i,1]:15} {cm[i,2]:15}")
    print()

    # Save training history
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump({
            'history': training_history,
            'best_val_acc': float(best_val_acc),
            'class_names': train_dataset.class_names,
            'num_train_samples': len(train_dataset),
            'num_val_samples': len(val_dataset)
        }, f, indent=2)

    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"Best model: {MODEL_DIR / 'best_model.pth'}")
    print(f"Training history: {MODEL_DIR / 'training_history.json'}")
    print()


if __name__ == "__main__":
    main()
