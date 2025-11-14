"""
Train EfficientNet-B3 on BALANCED Ultimate Dataset
Balances all 4 classes to 9,000 samples each with aggressive augmentation for rare classes

Target distribution:
- Asian hornets: 9,000 (sample from 30k)
- Bees: 9,000 (sample from 30k)
- European hornets: 9,000 (100 real × 90 augmented versions)
- Wasps: 9,000 (97 real × 93 augmented versions)

Total: 36,000 perfectly balanced samples
"""

import os
import sys
import torch

# Force unbuffered output
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import random


class BalancedUltimateDataset(Dataset):
    """
    Balanced dataset with aggressive augmentation for rare classes.
    Binary classification: Asian hornets vs Everything Else

    Strategy:
    - Common classes (Asian hornets, bees): Random sample 9k from 30k
    - Rare classes (European hornets, wasps): Apply 90x augmentation to reach 9k
    """

    def __init__(self, data_sources, split='train', standard_transform=None,  extreme_transform=None, target_per_class=9000):
        self.standard_transform = standard_transform
        self.extreme_transform = extreme_transform
        self.target_per_class = target_per_class

        # Store images with their class and augmentation needs
        self.samples = []  # List of (img_path, label, class_name, augmentation_transform)

        print(f"\nLoading {split.upper()} data with BALANCED sampling...")
        print(f"Target: {target_per_class} samples per class")

        # Track raw counts before augmentation
        raw_counts = {'asian_hornets': [], 'bees': [], 'european_hornets': [], 'wasps': []}

        # Load all images first
        if split == 'train':
            self._load_all_train_images(data_sources, raw_counts)

            # Now balance the dataset
            print(f"\nRaw counts:")
            print(f"  Asian hornets: {len(raw_counts['asian_hornets'])}")
            print(f"  Bees: {len(raw_counts['bees'])}")
            print(f"  European hornets: {len(raw_counts['european_hornets'])}")
            print(f"  Wasps: {len(raw_counts['wasps'])}")

            self._balance_dataset(raw_counts)

            print(f"\nBalanced dataset:")
            asian_count = sum(1 for s in self.samples if s[2] == 'asian_hornets')
            bee_count = sum(1 for s in self.samples if s[2] == 'bees')
            euro_count = sum(1 for s in self.samples if s[2] == 'european_hornets')
            wasp_count = sum(1 for s in self.samples if s[2] == 'wasps')

            print(f"  Asian hornets: {asian_count}")
            print(f"  Bees: {bee_count}")
            print(f"  European hornets: {euro_count}")
            print(f"  Wasps: {wasp_count}")
            print(f"  Total: {len(self.samples)}")
            print(f"  Class balance: {sum(1 for s in self.samples if s[1] == 1)/len(self.samples)*100:.1f}% positive")

        else:  # Validation - no augmentation, use original data
            self._load_validation_data(data_sources)
            print(f"  Total validation samples: {len(self.samples)}")

    def _load_all_train_images(self, data_sources, raw_counts):
        """Load all training images from YOLO datasets"""
        for dataset_key in ['bees_hornets1', 'bees_hornets2']:
            if dataset_key not in data_sources:
                continue

            data_root = Path(data_sources[dataset_key])
            images_dir = data_root / 'train' / 'images'
            labels_dir = data_root / 'train' / 'labels'

            if not images_dir.exists():
                continue

            for label_file in labels_dir.glob('*.txt'):
                image_name = label_file.stem + '.jpg'
                image_path = images_dir / image_name

                if not image_path.exists():
                    image_name = label_file.stem + '.png'
                    image_path = images_dir / image_name

                if not image_path.exists():
                    continue

                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        class_id = int(parts[0])
                        if class_id == 0:  # Bee
                            raw_counts['bees'].append(str(image_path))
                        elif class_id == 1:  # Asian Hornet
                            raw_counts['asian_hornets'].append(str(image_path))

        # Load European hornets and wasps from archive
        if 'archive' in data_sources:
            archive_root = Path(data_sources['archive']) / 'data3000' / 'data' / 'train' / 'images'

            if archive_root.exists():
                # European hornets
                euro_dir = archive_root / 'Vespa_crabro'
                if euro_dir.exists():
                    raw_counts['european_hornets'] = [str(p) for p in euro_dir.glob('*.jpg')]

                # Wasps
                wasp_dir = archive_root / 'Vespula_sp'
                if wasp_dir.exists():
                    raw_counts['wasps'] = [str(p) for p in wasp_dir.glob('*.jpg')]

    def _balance_dataset(self, raw_counts):
        """Balance dataset by sampling/augmenting to target_per_class"""

        # Asian hornets: Sample 9k from available
        asian_images = raw_counts['asian_hornets']
        if len(asian_images) > self.target_per_class:
            asian_images = random.sample(asian_images, self.target_per_class)

        for img_path in asian_images:
            self.samples.append((img_path, 1, 'asian_hornets', self.standard_transform))

        # Bees: Sample 9k from available
        bee_images = raw_counts['bees']
        if len(bee_images) > self.target_per_class:
            bee_images = random.sample(bee_images, self.target_per_class)

        for img_path in bee_images:
            self.samples.append((img_path, 0, 'bees', self.standard_transform))

        # European hornets: Augment heavily
        euro_images = raw_counts['european_hornets']
        if len(euro_images) > 0:
            augmentations_per_image = self.target_per_class // len(euro_images)
            remainder = self.target_per_class % len(euro_images)

            for img_path in euro_images:
                # Add multiple augmented versions
                for _ in range(augmentations_per_image):
                    self.samples.append((img_path, 0, 'european_hornets', self.extreme_transform))

            # Add remainder
            for img_path in random.sample(euro_images, min(remainder, len(euro_images))):
                self.samples.append((img_path, 0, 'european_hornets', self.extreme_transform))

        # Wasps: Augment heavily
        wasp_images = raw_counts['wasps']
        if len(wasp_images) > 0:
            augmentations_per_image = self.target_per_class // len(wasp_images)
            remainder = self.target_per_class % len(wasp_images)

            for img_path in wasp_images:
                # Add multiple augmented versions
                for _ in range(augmentations_per_image):
                    self.samples.append((img_path, 0, 'wasps', self.extreme_transform))

            # Add remainder
            for img_path in random.sample(wasp_images, min(remainder, len(wasp_images))):
                self.samples.append((img_path, 0, 'wasps', self.extreme_transform))

    def _load_validation_data(self, data_sources):
        """Load validation data (no augmentation)"""
        # Load from YOLO datasets
        for dataset_key in ['bees_hornets1', 'bees_hornets2']:
            if dataset_key not in data_sources:
                continue

            data_root = Path(data_sources[dataset_key])
            images_dir = data_root / 'valid' / 'images'
            labels_dir = data_root / 'valid' / 'labels'

            if not images_dir.exists():
                continue

            for label_file in labels_dir.glob('*.txt'):
                image_name = label_file.stem + '.jpg'
                image_path = images_dir / image_name

                if not image_path.exists():
                    image_name = label_file.stem + '.png'
                    image_path = images_dir / image_name

                if not image_path.exists():
                    continue

                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        class_id = int(parts[0])
                        if class_id == 0:  # Bee
                            self.samples.append((str(image_path), 0, 'bees', self.standard_transform))
                        elif class_id == 1:  # Asian Hornet
                            self.samples.append((str(image_path), 1, 'asian_hornets', self.standard_transform))

        # Load European hornets and wasps from archive
        if 'archive' in data_sources:
            archive_root = Path(data_sources['archive']) / 'data3000' / 'data' / 'val' / 'images'

            if archive_root.exists():
                # Asian hornets
                asian_dir = archive_root / 'Vespa_velutina'
                if asian_dir.exists():
                    for img_path in asian_dir.glob('*.jpg'):
                        self.samples.append((str(img_path), 1, 'asian_hornets', self.standard_transform))

                # European hornets
                euro_dir = archive_root / 'Vespa_crabro'
                if euro_dir.exists():
                    for img_path in euro_dir.glob('*.jpg'):
                        self.samples.append((str(img_path), 0, 'european_hornets', self.standard_transform))

                # Wasps
                wasp_dir = archive_root / 'Vespula_sp'
                if wasp_dir.exists():
                    for img_path in wasp_dir.glob('*.jpg'):
                        self.samples.append((str(img_path), 0, 'wasps', self.standard_transform))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name, transform = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if transform:
            image = transform(image)

        return image, label


def create_efficientnet_model(pretrained=True):
    """Create EfficientNet-B3 model for binary classification"""
    model = models.efficientnet_b3(pretrained=pretrained)

    # Unfreeze last 3 blocks for fine-tuning
    total_blocks = len(model.features)

    for i, block in enumerate(model.features):
        if i < total_blocks - 3:
            for param in block.parameters():
                param.requires_grad = False
        else:
            for param in block.parameters():
                param.requires_grad = True

    # Replace classifier for binary classification
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 1)
    )

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_preds.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    print("=" * 70)
    print("TRAIN EFFICIENTNET-B3 ON BALANCED ULTIMATE DATASET")
    print("=" * 70)
    print("Strategy: 9,000 samples per class with aggressive augmentation")
    print()

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    ULTIMATE_ROOT = Path(r"D:\Ultimate Dataset")
    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    data_sources = {
        'bees_hornets1': ULTIMATE_ROOT / "BeesAndHornets1" / "Bee And Asian Hornet Detection",
        'bees_hornets2': ULTIMATE_ROOT / "BeesAndHornets2" / "Dataset",
        'archive': ULTIMATE_ROOT / "archive"
    }

    # Training parameters
    BATCH_SIZE = 64  # Larger batch size for GPU
    NUM_EPOCHS = 15  # More epochs for balanced dataset
    LEARNING_RATE = 0.0001
    TARGET_PER_CLASS = 9000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # Data transforms
    print("Setting up augmentation transforms...")

    # Standard augmentation for common classes (Asian hornets, bees)
    standard_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # EXTREME augmentation for rare classes (European hornets, wasps)
    extreme_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # Very aggressive crops
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),  # More flips
        transforms.RandomRotation(45),  # Full rotation
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),  # Strong color variation
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),  # Perspective distortion
        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.3)),  # Affine transforms
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("Loading balanced training dataset...")
    train_dataset = BalancedUltimateDataset(
        data_sources,
        split='train',
        standard_transform=standard_transform,
        extreme_transform=extreme_transform,
        target_per_class=TARGET_PER_CLASS
    )

    print("\nLoading validation dataset...")
    val_dataset = BalancedUltimateDataset(
        data_sources,
        split='val',
        standard_transform=val_transform,
        extreme_transform=None,
        target_per_class=None
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create model
    print("\nCreating EfficientNet-B3 model...")
    model = create_efficientnet_model(pretrained=True)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Samples per class: {TARGET_PER_CLASS}")
    print()

    best_val_acc = 0.0
    best_recall = 0.0
    training_history = []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )

        # Learning rate scheduling
        scheduler.step(val_acc)

        epoch_time = time.time() - epoch_start

        # Calculate detailed metrics
        asian_mask = val_labels == 1
        negative_mask = val_labels == 0

        asian_recall = (val_preds[asian_mask] == 1).sum() / asian_mask.sum() if asian_mask.sum() > 0 else 0
        negative_acc = (val_preds[negative_mask] == 0).sum() / negative_mask.sum() if negative_mask.sum() > 0 else 0

        # Print progress
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s)", flush=True)
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%", flush=True)
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%", flush=True)
        print(f"  Asian Hornet Recall: {asian_recall*100:.2f}%", flush=True)
        print(f"  Negative Class Acc: {negative_acc*100:.2f}%", flush=True)

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'asian_recall': float(asian_recall),
            'negative_accuracy': float(negative_acc),
            'time': float(epoch_time)
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_recall = asian_recall
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'asian_recall': asian_recall,
            }, MODEL_DIR / 'best_balanced_efficientnet.pth')
            print(f"  [NEW BEST MODEL SAVED - Val Acc: {val_acc:.2f}%]", flush=True)
        print()

    total_time = time.time() - start_time

    # Final evaluation
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best Asian Hornet recall: {best_recall*100:.2f}%")

    # Load best model
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(MODEL_DIR / 'best_balanced_efficientnet.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, final_preds, final_labels, final_probs = validate(
        model, val_loader, criterion, device
    )

    # Classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(
        final_labels,
        final_preds,
        target_names=['NOT Asian Hornet', 'Asian Hornet'],
        digits=4
    ))

    # Confusion matrix
    print("CONFUSION MATRIX")
    print("=" * 70)
    cm = confusion_matrix(final_labels, final_preds)
    print(f"{'':25} {'NOT Asian Hornet':20} {'Asian Hornet':20}")
    print(f"{'NOT Asian Hornet':25} {cm[0,0]:20} {cm[0,1]:20}")
    print(f"{'Asian Hornet':25} {cm[1,0]:20} {cm[1,1]:20}")

    # Calculate metrics
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\nDetailed Metrics:")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall: {recall*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")

    # ROC AUC
    fpr_curve, tpr_curve, thresholds = roc_curve(final_labels, final_probs)
    roc_auc = auc(fpr_curve, tpr_curve)
    print(f"  ROC AUC Score: {roc_auc:.4f}")

    # Save results
    results = {
        'history': training_history,
        'best_val_acc': float(best_val_acc),
        'best_recall': float(best_recall),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'false_positive_rate': float(fpr),
        'num_train_samples': len(train_dataset),
        'num_val_samples': len(val_dataset),
        'model': 'EfficientNet-B3',
        'dataset': 'Balanced Ultimate Dataset (9k per class)',
        'target_per_class': TARGET_PER_CLASS
    }

    with open(MODEL_DIR / 'balanced_efficientnet_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"Best model: {MODEL_DIR / 'best_balanced_efficientnet.pth'}")
    print(f"Training results: {MODEL_DIR / 'balanced_efficientnet_results.json'}")
    print()


if __name__ == "__main__":
    main()
