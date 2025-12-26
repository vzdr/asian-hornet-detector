"""
Multi-Class Asian Hornet Classifier Training Script
====================================================

4-class classification:
- Class 0: Asian Hornets (~45k images)
- Class 1: Bees (~45k images)
- Class 2: European Hornets (~38k images)
- Class 3: Wasps (~31k images)

This script trains an EfficientNet-B3 model with:
- CrossEntropyLoss (multi-class classification)
- Softmax activation for natural confidence calibration
- Balanced sampling (30k per class)
- Data augmentation for generalization

Key Improvements over Binary Model:
- Natural confidence scores via softmax (not 0%/100%)
- Can distinguish between all 4 species
- Shows uncertainty for borderline cases
"""

import os
import sys
import torch

# Force unbuffered output for real-time monitoring
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import random


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class MultiClassDataset(Dataset):
    """
    Dataset for 4-class hornet/bee/wasp classification
    Loads from YOLO datasets (Asian hornets + bees) and GBIF (European hornets + wasps)
    """

    def __init__(self, data_sources, split='train', transform=None, samples_per_class=None):
        """
        Args:
            data_sources: dict with paths to data sources
            split: 'train' or 'valid'
            transform: image transformations
            samples_per_class: if set, sample this many images per class (for balanced training)
        """
        self.transform = transform
        self.samples = []  # List of (img_path, label, class_name)

        # Class mapping
        self.class_to_idx = {
            'asian_hornets': 0,
            'bees': 1,
            'european_hornets': 2,
            'wasps': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.class_names = ['Asian Hornet', 'Bee', 'European Hornet', 'Wasp']

        print(f"\nLoading {split.upper()} data...")

        # Collect all images by class
        class_images = {class_name: [] for class_name in self.class_to_idx.keys()}

        # 1. Load Asian hornets and Bees from YOLO datasets
        self._load_yolo_data(data_sources, split, class_images)

        # 2. Load European hornets from GBIF
        self._load_gbif_european_hornets(data_sources, class_images)

        # 3. Load Wasps from GBIF
        self._load_gbif_wasps(data_sources, class_images)

        # Print raw counts
        print("\nRaw image counts:")
        for class_name, images in class_images.items():
            print(f"  {class_name}: {len(images)}")

        # Sample if requested (for balanced training)
        if samples_per_class and split == 'train':
            print(f"\nBalancing dataset: {samples_per_class} images per class...")
            for class_name, images in class_images.items():
                if len(images) > samples_per_class:
                    sampled = random.sample(images, samples_per_class)
                else:
                    sampled = images
                    print(f"  Warning: {class_name} has only {len(images)} images (< {samples_per_class})")

                label = self.class_to_idx[class_name]
                for img_path in sampled:
                    self.samples.append((img_path, label, class_name))
        else:
            # Use all available images
            for class_name, images in class_images.items():
                label = self.class_to_idx[class_name]
                for img_path in images:
                    self.samples.append((img_path, label, class_name))

        # Shuffle samples
        random.shuffle(self.samples)

        print(f"\nFinal dataset size: {len(self.samples)}")
        for class_name in self.class_to_idx.keys():
            count = sum(1 for s in self.samples if s[2] == class_name)
            print(f"  {class_name}: {count}")

    def _load_yolo_data(self, data_sources, split, class_images):
        """Load Asian hornets and bees from YOLO format datasets"""
        for dataset_key in ['bees_hornets1', 'bees_hornets2']:
            if dataset_key not in data_sources:
                continue

            data_root = Path(data_sources[dataset_key])

            # Map split names
            yolo_split = split
            if split == 'valid' and not (data_root / 'valid').exists():
                if (data_root / 'val').exists():
                    yolo_split = 'val'

            images_dir = data_root / yolo_split / 'images'
            labels_dir = data_root / yolo_split / 'labels'

            if not images_dir.exists() or not labels_dir.exists():
                print(f"  Skipping {dataset_key}/{yolo_split} (directory not found)")
                continue

            # Process YOLO labels
            for label_file in labels_dir.glob('*.txt'):
                image_name = label_file.stem + '.jpg'
                image_path = images_dir / image_name

                if not image_path.exists():
                    image_name = label_file.stem + '.png'
                    image_path = images_dir / image_name

                if not image_path.exists():
                    continue

                # Read YOLO label (class x y w h)
                try:
                    with open(label_file, 'r') as f:
                        first_line = f.readline().strip()

                    if not first_line:
                        continue

                    parts = first_line.split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])

                    # Class 0 = Bee, Class 1 = Asian Hornet (from YOLO data.yaml)
                    if class_id == 0:
                        class_images['bees'].append(str(image_path))
                    elif class_id == 1:
                        class_images['asian_hornets'].append(str(image_path))
                except Exception as e:
                    continue

    def _load_gbif_european_hornets(self, data_sources, class_images):
        """Load European hornets from GBIF download"""
        if 'gbif_european_hornets' not in data_sources:
            return

        gbif_dir = Path(data_sources['gbif_european_hornets'])
        if not gbif_dir.exists():
            print(f"  Warning: GBIF European hornets directory not found: {gbif_dir}")
            return

        # Load all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPG', '.JPEG', '.PNG'}
        for img_path in gbif_dir.iterdir():
            if img_path.is_file() and img_path.suffix in image_extensions:
                class_images['european_hornets'].append(str(img_path))

    def _load_gbif_wasps(self, data_sources, class_images):
        """Load wasps from GBIF download"""
        if 'gbif_wasps' not in data_sources:
            return

        gbif_dir = Path(data_sources['gbif_wasps'])
        if not gbif_dir.exists():
            print(f"  Warning: GBIF wasps directory not found: {gbif_dir}")
            return

        # Load all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPG', '.JPEG', '.PNG'}
        for img_path in gbif_dir.iterdir():
            if img_path.is_file() and img_path.suffix in image_extensions:
                class_images['wasps'].append(str(img_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {running_loss/(batch_idx+1):.4f} '
                  f'Acc: {100.*correct/total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * accuracy_score(all_labels, all_preds)

    return val_loss, val_acc, all_preds, all_labels


def main():
    print("="*70)
    print("MULTI-CLASS ASIAN HORNET CLASSIFIER TRAINING")
    print("="*70)
    print("\n4-Class Classification:")
    print("  0: Asian Hornets")
    print("  1: Bees")
    print("  2: European Hornets")
    print("  3: Wasps")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data sources
    data_sources = {
        'bees_hornets1': r'D:\Ultimate Dataset\BeesAndHornets1\Bee And Asian Hornet Detection',
        'bees_hornets2': r'D:\Ultimate Dataset\BeesAndHornets2\Dataset',
        'gbif_european_hornets': r'D:\Ultimate Dataset\european_hornets_gbif',
        'gbif_wasps': r'D:\Ultimate Dataset\wasps_gbif'
    }

    # Hyperparameters
    BATCH_SIZE = 64  # Increase if GPU memory allows
    LEARNING_RATE = 0.0001  # Lower for fine-tuning
    EPOCHS = 15
    NUM_WORKERS = 4
    SAMPLES_PER_CLASS = 30000  # 30k per class = 120k total

    print(f"\nHyperparameters:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Samples per class: {SAMPLES_PER_CLASS}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)

    train_dataset = MultiClassDataset(
        data_sources=data_sources,
        split='train',
        transform=train_transform,
        samples_per_class=SAMPLES_PER_CLASS
    )

    val_dataset = MultiClassDataset(
        data_sources=data_sources,
        split='valid',
        transform=val_transform,
        samples_per_class=None  # Use all validation data
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Model
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)

    model = models.efficientnet_b3(pretrained=True)
    print("Loaded EfficientNet-B3 pretrained on ImageNet")

    # Freeze base layers, unfreeze last 3 blocks for fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 blocks
    for block in list(model.features)[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    # Modify classifier for 4 classes
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 4)
    )

    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model: EfficientNet-B3 (4 classes)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Create output directory
    output_dir = Path('multiclass_models')
    output_dir.mkdir(exist_ok=True)

    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"{'='*70}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_multiclass_model.pth')
            print(f"  *** New best model saved! Val Acc: {val_acc:.2f}% ***")

    # Training time
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time/60:.1f} minutes")

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    model.load_state_dict(torch.load(output_dir / 'best_multiclass_model.pth'))
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

    print(f"\nBest Validation Accuracy: {val_acc:.2f}%")

    # Classification report
    class_names = ['Asian Hornet', 'Bee', 'European Hornet', 'Wasp']
    print("\nPer-Class Performance:")
    print(classification_report(val_labels, val_preds, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    print("\nConfusion Matrix:")
    print("Rows=Actual, Cols=Predicted")
    print(f"{'':20} {' '.join([f'{c:>15}' for c in class_names])}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:20} {' '.join([f'{v:>15}' for v in row])}")

    # Save results
    results = {
        'best_val_acc': float(val_acc),
        'training_time_minutes': training_time / 60,
        'classification_report': classification_report(val_labels, val_preds, target_names=class_names, output_dict=True),
        'confusion_matrix': cm.tolist(),
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'samples_per_class': SAMPLES_PER_CLASS
        }
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model saved to: {output_dir / 'best_multiclass_model.pth'}")
    print(f"Results saved to: {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
