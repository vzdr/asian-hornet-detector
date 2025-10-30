#  EfficientNet-B3 binary

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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np


class BinaryHornetDataset(Dataset):
    # Binary dataset: Asian hornet (positive) vs Everything else (negative)
    def __init__(self, hornet_root, random_root, split='train', transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        # POSITIVE: Asian hornets
        asian_hornet_dir = Path(hornet_root) / split / 'images' / 'Vespa_velutina'
        if asian_hornet_dir.exists():
            for img_path in asian_hornet_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(1)

        # NEGATIVE: European hornets
        european_hornet_dir = Path(hornet_root) / split / 'images' / 'Vespa_crabro'
        if european_hornet_dir.exists():
            for img_path in european_hornet_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(0)

        # NEGATIVE: Wasps
        wasp_dir = Path(hornet_root) / split / 'images' / 'Vespula_sp'
        if wasp_dir.exists():
            for img_path in wasp_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(0)

        # NEGATIVE: Random images (training only, use 2000 instead of 1500)
        if split == 'train' and random_root:
            random_dir = Path(random_root) / 'data'
            if random_dir.exists():
                for img_path in list(random_dir.glob('*.jpg'))[:2000]:
                    self.images.append(img_path)
                    self.labels.append(0)

        print(f"\n{split.upper()} SET:")
        print(f"  Asian hornets (POSITIVE): {sum(self.labels)}")
        print(f"  NOT Asian hornets (NEGATIVE): {len(self.labels) - sum(self.labels)}")
        print(f"  Total: {len(self.labels)}")
        print(f"  Class balance: {sum(self.labels)/len(self.labels)*100:.1f}% positive")

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
    
    # Load pre-trained EfficientNet-B3
    model = models.efficientnet_b3(pretrained=pretrained)

    # Unfreeze the last few blocks for fine tuning, I think this is enough, its what I found people mostly do
    # EfficientNet-B3 has 'features' and 'classifier'
    # We'll unfreeze the last 3 blocks of features + classifier
    total_blocks = len(model.features)

    # Freeze early blocks
    for i, block in enumerate(model.features):
        if i < total_blocks - 3:  # Freeze all except last 3 blocks
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
    print("=" * 70) #formatting, easier for me to see the results
    print("IMPROVED BINARY ASIAN HORNET DETECTOR - EfficientNet-B3")
    print("=" * 70)

    # Configuration
    HORNET_DATA_ROOT = Path(r"C:\Users\Zdravkovic\Downloads\archive\data3000\data")
    RANDOM_DATA_ROOT = Path(r"C:\Users\Zdravkovic\Desktop\randomimages")
    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    BATCH_SIZE = 24  # Smaller batch for EfficientNet (more memory intensive)
    NUM_EPOCHS = 25  # More epochs
    LEARNING_RATE = 0.0001  # Lower LR since we're unfreezing more layers

    device = torch.device('cpu')
    print(f"\nDevice: {device}")

    # STRONGER Data transforms
    print("\nSetting up STRONGER data augmentation...")
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Slightly larger
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # Add vertical flip
        transforms.RandomRotation(20),  # More rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stronger
        transforms.RandomGrayscale(p=0.1),  # Occasionally grayscale
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
    train_dataset = BinaryHornetDataset(
        HORNET_DATA_ROOT,
        RANDOM_DATA_ROOT,
        split='train',
        transform=train_transform
    )
    val_dataset = BinaryHornetDataset(
        HORNET_DATA_ROOT,
        None,
        split='val',
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create EfficientNet-B3 model
    print("\nCreating EfficientNet-B3 model...")
    print("Loading pre-trained weights (this may take a moment)...")
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
    print(f"Model: EfficientNet-B3")
    print(f"Strategy: Fine-tune last 3 blocks + classifier")
    print()

    best_val_acc = 0.0
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

        # Print progress
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.2f}%")

        # Calculate recall for Asian hornets (most important metric)
        asian_hornet_mask = val_labels == 1
        if asian_hornet_mask.sum() > 0:
            asian_recall = (val_preds[asian_hornet_mask] == 1).sum() / asian_hornet_mask.sum()
            print(f"  Asian Hornet Recall: {asian_recall*100:.2f}%")

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
            }, MODEL_DIR / 'best_binary_efficientnet.pth')
            print(f"  [NEW BEST MODEL SAVED]")
        print()

    total_time = time.time() - start_time

    # Final evaluation
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Load best model for detailed evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(MODEL_DIR / 'best_binary_efficientnet.pth')
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
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Asian Hornet Detection:")
    print(f"  True Positives: {tp} / {tp + fn} ({recall*100:.2f}% recall)")
    print(f"  False Negatives: {fn} (missed {fn} Asian hornets)")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  F1-Score: {f1*100:.2f}%")
    print()

    # ROC AUC
    fpr, tpr, thresholds = roc_curve(final_labels, final_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Find optimal threshold
    from sklearn.metrics import f1_score
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.3, 0.9, 0.05):
        preds_at_threshold = (final_probs > threshold).astype(int)
        f1_thresh = f1_score(final_labels, preds_at_threshold)
        if f1_thresh > best_f1:
            best_f1 = f1_thresh
            best_threshold = threshold

    print(f"\nOptimal threshold (max F1): {best_threshold:.2f}")
    print(f"F1-score at optimal threshold: {best_f1:.4f}")

    # Save results
    results = {
        'history': training_history,
        'best_val_acc': float(best_val_acc),
        'roc_auc': float(roc_auc),
        'optimal_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'confusion_matrix': cm.tolist(),
        'precision': float(precision),
        'recall': float(recall),
        'num_train_samples': len(train_dataset),
        'num_val_samples': len(val_dataset),
        'model': 'EfficientNet-B3'
    }

    with open(MODEL_DIR / 'binary_efficientnet_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"Best model: {MODEL_DIR / 'best_binary_efficientnet.pth'}")
    print(f"Training results: {MODEL_DIR / 'binary_efficientnet_results.json'}")
    print()


if __name__ == "__main__":
    main()
