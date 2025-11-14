"""
Retrain EfficientNet-B3 on Ultimate Dataset with Bees Included
Binary Classification: Asian Hornets vs Everything Else (bees, European hornets, wasps)
"""

import os
import sys
import torch

# Force unbuffered output so we can see progress immediately
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


class UltimateDataset(Dataset):
    """
    Binary dataset using Ultimate Dataset structure:
    POSITIVE (1): Asian hornets
    NEGATIVE (0): Bees, European hornets, Wasps
    """
    def __init__(self, data_sources, split='train', transform=None, max_samples_per_class=None):
        self.transform = transform
        self.images = []
        self.labels = []

        print(f"\nLoading {split.upper()} data...")

        # Track counts
        counts = {'asian_hornets': 0, 'bees': 0, 'european_hornets': 0, 'wasps': 0}

        # Parse BeesAndHornets1 (YOLO format)
        if 'bees_hornets1' in data_sources:
            self._load_yolo_dataset(
                data_sources['bees_hornets1'],
                split,
                counts,
                max_samples_per_class
            )

        # Parse BeesAndHornets2 (YOLO format)
        if 'bees_hornets2' in data_sources:
            self._load_yolo_dataset(
                data_sources['bees_hornets2'],
                split,
                counts,
                max_samples_per_class
            )

        # Parse archive dataset (class folders)
        if 'archive' in data_sources and split == 'val':
            self._load_archive_dataset(
                data_sources['archive'],
                counts,
                max_samples_per_class
            )

        # Print statistics
        print(f"  Asian hornets (POSITIVE): {counts['asian_hornets']}")
        print(f"  Bees (NEGATIVE): {counts['bees']}")
        print(f"  European hornets (NEGATIVE): {counts['european_hornets']}")
        print(f"  Wasps (NEGATIVE): {counts['wasps']}")
        print(f"  Total: {len(self.labels)}")
        print(f"  Class balance: {sum(self.labels)/len(self.labels)*100:.1f}% positive")

    def _load_yolo_dataset(self, data_root, split, counts, max_samples):
        """Load data from YOLO format datasets (BeesAndHornets1/2)"""
        data_root = Path(data_root)

        # Map split names
        split_map = {'train': 'train', 'val': 'valid'}
        yolo_split = split_map.get(split, split)

        images_dir = data_root / yolo_split / 'images'
        labels_dir = data_root / yolo_split / 'labels'

        if not images_dir.exists():
            print(f"  Skipping {data_root.name}/{yolo_split} (not found)")
            return

        # Read all label files
        asian_images = []
        bee_images = []

        for label_file in labels_dir.glob('*.txt'):
            image_name = label_file.stem + '.jpg'
            image_path = images_dir / image_name

            if not image_path.exists():
                image_name = label_file.stem + '.png'
                image_path = images_dir / image_name

            if not image_path.exists():
                continue

            # Read label (YOLO: class x y w h)
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])

                if class_id == 0:  # Bee
                    bee_images.append(str(image_path))
                elif class_id == 1:  # Asian Hornet
                    asian_images.append(str(image_path))

        # Apply max_samples limit
        if max_samples:
            asian_images = random.sample(asian_images, min(max_samples, len(asian_images)))
            bee_images = random.sample(bee_images, min(max_samples, len(bee_images)))

        # Add to dataset
        for img_path in asian_images:
            self.images.append(img_path)
            self.labels.append(1)
            counts['asian_hornets'] += 1

        for img_path in bee_images:
            self.images.append(img_path)
            self.labels.append(0)
            counts['bees'] += 1

    def _load_archive_dataset(self, data_root, counts, max_samples):
        """Load data from archive dataset (class folders)"""
        data_root = Path(data_root) / 'data3000' / 'data' / 'val' / 'images'

        if not data_root.exists():
            return

        # Asian hornets
        asian_dir = data_root / 'Vespa_velutina'
        if asian_dir.exists():
            asian_images = list(asian_dir.glob('*.jpg'))
            if max_samples:
                asian_images = random.sample(asian_images, min(max_samples, len(asian_images)))

            for img_path in asian_images:
                self.images.append(str(img_path))
                self.labels.append(1)
                counts['asian_hornets'] += 1

        # European hornets
        euro_dir = data_root / 'Vespa_crabro'
        if euro_dir.exists():
            euro_images = list(euro_dir.glob('*.jpg'))
            if max_samples:
                euro_images = random.sample(euro_images, min(max_samples, len(euro_images)))

            for img_path in euro_images:
                self.images.append(str(img_path))
                self.labels.append(0)
                counts['european_hornets'] += 1

        # Wasps
        wasp_dir = data_root / 'Vespula_sp'
        if wasp_dir.exists():
            wasp_images = list(wasp_dir.glob('*.jpg'))
            if max_samples:
                wasp_images = random.sample(wasp_images, min(max_samples, len(wasp_images)))

            for img_path in wasp_images:
                self.images.append(str(img_path))
                self.labels.append(0)
                counts['wasps'] += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

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
    print("RETRAIN EFFICIENTNET-B3 ON ULTIMATE DATASET WITH BEES")
    print("=" * 70)

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    ULTIMATE_ROOT = Path(r"D:\Ultimate Dataset")
    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    # Data sources
    data_sources = {
        'bees_hornets1': ULTIMATE_ROOT / "BeesAndHornets1" / "Bee And Asian Hornet Detection",
        'bees_hornets2': ULTIMATE_ROOT / "BeesAndHornets2" / "Dataset",
        'archive': ULTIMATE_ROOT / "archive"
    }

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10  # Reduced from 20 for faster training (Epoch 1 already got 99% recall!)
    LEARNING_RATE = 0.0001
    MAX_TRAIN_SAMPLES_PER_CLASS = 15000  # Limit to prevent imbalance

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data transforms
    print("\nSetting up data augmentation...")
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
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
    train_dataset = UltimateDataset(
        data_sources,
        split='train',
        transform=train_transform,
        max_samples_per_class=MAX_TRAIN_SAMPLES_PER_CLASS
    )
    val_dataset = UltimateDataset(
        data_sources,
        split='val',
        transform=val_transform,
        max_samples_per_class=None
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    print("\nCreating EfficientNet-B3 model...")
    print("Loading pre-trained weights...")
    model = create_efficientnet_model(pretrained=True)
    model = model.to(device)

    # Count trainable parameters
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
    print(f"Max samples per class (train): {MAX_TRAIN_SAMPLES_PER_CLASS}")
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
        asian_hornet_mask = val_labels == 1
        bee_mask = val_labels == 0

        if asian_hornet_mask.sum() > 0:
            asian_recall = (val_preds[asian_hornet_mask] == 1).sum() / asian_hornet_mask.sum()
            asian_precision = (val_preds[asian_hornet_mask] == 1).sum() / (val_preds == 1).sum() if (val_preds == 1).sum() > 0 else 0
        else:
            asian_recall = 0
            asian_precision = 0

        if bee_mask.sum() > 0:
            bee_accuracy = (val_preds[bee_mask] == 0).sum() / bee_mask.sum()
        else:
            bee_accuracy = 0

        # Print progress with explicit flushing
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s)", flush=True)
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%", flush=True)
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%", flush=True)
        print(f"  Asian Hornet - Recall: {asian_recall*100:.2f}%  Precision: {asian_precision*100:.2f}%", flush=True)
        print(f"  Bee Accuracy (NOT Asian Hornet): {bee_accuracy*100:.2f}%", flush=True)

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'asian_recall': float(asian_recall),
            'asian_precision': float(asian_precision),
            'bee_accuracy': float(bee_accuracy),
            'time': float(epoch_time)
        })

        # Save best model (prioritize recall for Asian hornets)
        if asian_recall > best_recall:
            best_recall = asian_recall
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'asian_recall': asian_recall,
            }, MODEL_DIR / 'best_ultimate_efficientnet.pth')
            print(f"  [NEW BEST MODEL SAVED - Recall: {asian_recall*100:.2f}%]")
        print()

    total_time = time.time() - start_time

    # Final evaluation
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best Asian Hornet recall: {best_recall*100:.2f}%")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model for detailed evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(MODEL_DIR / 'best_ultimate_efficientnet.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    _, _, final_preds, final_labels, final_probs = validate(
        model, val_loader, criterion, device
    )

    # Classification report
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION REPORT")
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
    print("Rows = Actual, Columns = Predicted")
    print(f"\n{'':25} {'NOT Asian Hornet':20} {'Asian Hornet':20}")
    print(f"{'NOT Asian Hornet':25} {cm[0,0]:20} {cm[0,1]:20}")
    print(f"{'Asian Hornet':25} {cm[1,0]:20} {cm[1,1]:20}")
    print()

    # Calculate metrics
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Detailed Metrics:")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall: {recall*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    print()

    # ROC AUC
    fpr_curve, tpr_curve, thresholds = roc_curve(final_labels, final_probs)
    roc_auc = auc(fpr_curve, tpr_curve)
    print(f"ROC AUC Score: {roc_auc:.4f}")

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
        'dataset': 'Ultimate Dataset with Bees'
    }

    with open(MODEL_DIR / 'ultimate_efficientnet_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"Best model: {MODEL_DIR / 'best_ultimate_efficientnet.pth'}")
    print(f"Training results: {MODEL_DIR / 'ultimate_efficientnet_results.json'}")
    print()


if __name__ == "__main__":
    main()
