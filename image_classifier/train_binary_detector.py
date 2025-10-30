"""
Binary Asian Hornet Detector
Training script: "Is this an Asian hornet? YES/NO"

Positive class: Vespa_velutina (Asian hornet)
Negative class: Everything else (European hornets, wasps, random images)
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np


class BinaryHornetDataset(Dataset):
    """
    Binary dataset: Asian hornet (positive) vs Everything else (negative)
    """
    def __init__(self, hornet_root, random_root, split='train', transform=None):
        self.transform = transform
        self.images = []
        self.labels = []  # 1 = Asian hornet, 0 = NOT Asian hornet

        # POSITIVE CLASS: Asian hornets only
        asian_hornet_dir = Path(hornet_root) / split / 'images' / 'Vespa_velutina'
        if asian_hornet_dir.exists():
            for img_path in asian_hornet_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(1)  # POSITIVE

        # NEGATIVE CLASS: European hornets
        european_hornet_dir = Path(hornet_root) / split / 'images' / 'Vespa_crabro'
        if european_hornet_dir.exists():
            for img_path in european_hornet_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(0)  # NEGATIVE

        # NEGATIVE CLASS: Wasps
        wasp_dir = Path(hornet_root) / split / 'images' / 'Vespula_sp'
        if wasp_dir.exists():
            for img_path in wasp_dir.glob('*.jpg'):
                self.images.append(img_path)
                self.labels.append(0)  # NEGATIVE

        # NEGATIVE CLASS: Random images (only for training split)
        if split == 'train' and random_root:
            random_dir = Path(random_root) / 'data'
            if random_dir.exists():
                for img_path in list(random_dir.glob('*.jpg'))[:1500]:  # Use 1500 random images
                    self.images.append(img_path)
                    self.labels.append(0)  # NEGATIVE

        print(f"\n{split.upper()} SET:")
        print(f"  Asian hornets (POSITIVE): {sum(self.labels)}")
        print(f"  NOT Asian hornets (NEGATIVE): {len(self.labels) - sum(self.labels)}")
        print(f"  Total: {len(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label


def create_binary_model(pretrained=True):
    """
    Create ResNet50 for binary classification.
    Output: Single probability (sigmoid) - probability of being Asian hornet
    """
    model = models.resnet50(pretrained=pretrained)

    # Freeze early layers for transfer learning
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Single output for binary classification

    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
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

        # Binary predictions
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

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
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Get probabilities
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
    print("BINARY ASIAN HORNET DETECTOR - TRAINING")
    print("=" * 70)

    # Configuration
    HORNET_DATA_ROOT = Path(r"C:\Users\Zdravkovic\Downloads\archive\data3000\data")
    RANDOM_DATA_ROOT = Path(r"C:\Users\Zdravkovic\Desktop\randomimages")
    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    BATCH_SIZE = 32
    NUM_EPOCHS = 15  # More epochs since binary is simpler
    LEARNING_RATE = 0.001

    device = torch.device('cpu')
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
    train_dataset = BinaryHornetDataset(
        HORNET_DATA_ROOT,
        RANDOM_DATA_ROOT,
        split='train',
        transform=train_transform
    )
    val_dataset = BinaryHornetDataset(
        HORNET_DATA_ROOT,
        None,  # Don't use random images for validation
        split='val',
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    print("\nCreating ResNet50 model for binary classification...")
    model = create_binary_model(pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    # Use BCEWithLogitsLoss (combines sigmoid + BCE for numerical stability)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
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
            }, MODEL_DIR / 'best_binary_model.pth')
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
    checkpoint = torch.load(MODEL_DIR / 'best_binary_model.pth')
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

    # Calculate ROC AUC
    fpr, tpr, thresholds = roc_curve(final_labels, final_probs)
    roc_auc = auc(fpr, tpr)

    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(final_labels, final_probs)

    print(f"ROC AUC Score: {roc_auc:.4f}")
    print()

    # Find optimal threshold (maximize F1)
    from sklearn.metrics import f1_score
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.3, 0.9, 0.05):
        preds_at_threshold = (final_probs > threshold).astype(int)
        f1 = f1_score(final_labels, preds_at_threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold (max F1): {best_threshold:.2f}")
    print(f"F1-score at optimal threshold: {best_f1:.4f}")

    # Save results
    results = {
        'history': training_history,
        'best_val_acc': float(best_val_acc),
        'roc_auc': float(roc_auc),
        'optimal_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'confusion_matrix': cm.tolist(),
        'num_train_samples': len(train_dataset),
        'num_val_samples': len(val_dataset),
        'num_positive_train': sum(train_dataset.labels),
        'num_positive_val': int(final_labels.sum())
    }

    with open(MODEL_DIR / 'binary_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(f"Best model: {MODEL_DIR / 'best_binary_model.pth'}")
    print(f"Training results: {MODEL_DIR / 'binary_training_results.json'}")
    print()


if __name__ == "__main__":
    main()
