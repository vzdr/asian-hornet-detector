"""
Binary Asian Hornet Classifier - Proper Implementation
Binary classification: Asian Hornet (1) vs Everything Else (0)
Using user specifications: EfficientNet-B3, frozen base with last 3 blocks unfrozen
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time
from tqdm import tqdm

# ============== HYPERPARAMETERS (User Specifications) ==============
BATCH_SIZE = 64  # Start with 64 as specified
LEARNING_RATE = 0.0001
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
IMG_SIZE = 224

print("="*70)
print("BINARY ASIAN HORNET CLASSIFIER")
print("="*70)
print(f"\nConfiguration:")
print(f"  Model: EfficientNet-B3 (frozen base + last 3 blocks unfrozen)")
print(f"  Classification: Binary (Asian Hornet vs Everything Else)")
print(f"  Loss: BCEWithLogitsLoss")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Device: {DEVICE}")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print()

# ============== DATASET CLASS ==============
class SimpleBinaryDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# ============== DATA TRANSFORMS ==============
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============== LOAD DATA ==============
print("Loading datasets...")

# Parse YOLO labels from BOTH datasets
print("\nParsing YOLO datasets...")

asian_hornet_images = []
bee_images = []

# Dataset 1: BeesAndHornets1
yolo1_images_dir = Path("D:/Ultimate Dataset/BeesAndHornets1/Bee And Asian Hornet Detection/train/images")
yolo1_labels_dir = Path("D:/Ultimate Dataset/BeesAndHornets1/Bee And Asian Hornet Detection/train/labels")

if yolo1_labels_dir.exists():
    print("  Loading BeesAndHornets1...")
    for label_file in yolo1_labels_dir.glob("*.txt"):
        try:
            with open(label_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    img_name = label_file.stem
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_path = yolo1_images_dir / f"{img_name}{ext}"
                        if img_path.exists():
                            if class_id == 1:
                                asian_hornet_images.append(img_path)
                            elif class_id == 0:
                                bee_images.append(img_path)
                            break
        except:
            continue

# Dataset 2: BeesAndHornets2
yolo2_images_dir = Path("D:/Ultimate Dataset/BeesAndHornets2/Dataset/train/images")
yolo2_labels_dir = Path("D:/Ultimate Dataset/BeesAndHornets2/Dataset/train/labels")

if yolo2_labels_dir.exists():
    print("  Loading BeesAndHornets2...")
    for label_file in yolo2_labels_dir.glob("*.txt"):
        try:
            with open(label_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    img_name = label_file.stem
                    for ext in ['.jpg', '.jpeg', '.png']:
                        img_path = yolo2_images_dir / f"{img_name}{ext}"
                        if img_path.exists():
                            if class_id == 1:
                                asian_hornet_images.append(img_path)
                            elif class_id == 0:
                                bee_images.append(img_path)
                            break
        except:
            continue

print(f"\n  Total Asian Hornets: {len(asian_hornet_images)} images")
print(f"  Total Bees: {len(bee_images)} images")

# Load GBIF datasets
european_hornet_dir = Path("D:/Ultimate Dataset/european_hornets_gbif")
wasp_dir = Path("D:/Ultimate Dataset/wasps_gbif")

european_hornet_images = list(european_hornet_dir.glob("*.jpg")) + list(european_hornet_dir.glob("*.jpeg")) + list(european_hornet_dir.glob("*.png"))
wasp_images = list(wasp_dir.glob("*.jpg")) + list(wasp_dir.glob("*.jpeg")) + list(wasp_dir.glob("*.png"))

print(f"  European Hornets (GBIF): {len(european_hornet_images)} images")
print(f"  Wasps (GBIF): {len(wasp_images)} images")

# Create dataset using file lists instead of directories
print("\nCreating training dataset...")

# Build samples: Asian hornets (positive class = 1), everything else (negative class = 0)
all_samples = []

# Use ALL Asian hornets (positive class)
print(f"Using ALL {len(asian_hornet_images)} Asian hornet images (positive class)")
for img_path in asian_hornet_images:
    all_samples.append((str(img_path), 1))

# Combine all negative class images
negative_images = bee_images + european_hornet_images + wasp_images
print(f"Total negative class images available: {len(negative_images)}")
print(f"  Bees: {len(bee_images)}")
print(f"  European Hornets: {len(european_hornet_images)}")
print(f"  Wasps: {len(wasp_images)}")

# Balance: use same number of negative samples as positive samples
np.random.shuffle(negative_images)
negative_samples = negative_images[:len(asian_hornet_images)]

for img_path in negative_samples:
    all_samples.append((str(img_path), 0))

print(f"\nBalanced dataset:")
print(f"  Positive (Asian Hornets): {len(asian_hornet_images)} images")
print(f"  Negative (Bees + European Hornets + Wasps): {len(negative_samples)} images")
print(f"  Unused negative images: {len(negative_images) - len(negative_samples)}")

# Shuffle all samples
np.random.shuffle(all_samples)

train_dataset = SimpleBinaryDataset(all_samples, transform=train_transform)
print(f"\nTotal dataset size: {len(train_dataset)} images")

# Use 10% of training data for validation
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

print(f"\nDataset split:")
print(f"  Training: {len(train_dataset)} samples")
print(f"  Validation: {len(val_dataset)} samples")

# Try to create dataloaders with specified batch size
try:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"\nDataLoaders created with batch size: {BATCH_SIZE}")
except:
    print(f"\nWARNING: Could not create DataLoader with batch size {BATCH_SIZE}")
    print("Falling back to batch size 32...")
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ============== MODEL SETUP ==============
print("\nSetting up model...")

# Load pretrained EfficientNet-B3
model = models.efficientnet_b3(pretrained=True)

# Freeze all base layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 3 blocks of EfficientNet-B3
# EfficientNet-B3 has features organized as Sequential with blocks
# Last 3 blocks are typically blocks 6, 7, and the final conv layer
print("Unfreezing last 3 blocks...")
total_blocks = len(model.features)
unfreeze_from = max(0, total_blocks - 3)

for i in range(unfreeze_from, total_blocks):
    for param in model.features[i].parameters():
        param.requires_grad = True

# Replace classifier with single output for binary classification
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 1)  # Single output for binary

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters: {total_params - trainable_params:,}")

model = model.to(DEVICE)

# ============== LOSS AND OPTIMIZER ==============
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

print(f"\nOptimizer: Adam (lr={LEARNING_RATE})")
print(f"Loss: BCEWithLogitsLoss")
print(f"Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")

# ============== TRAINING LOOP ==============
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # Shape: (batch_size,)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels

# ============== TRAINING ==============
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

best_val_loss = float('inf')
best_val_acc = 0
training_start = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 70)

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, DEVICE)

    epoch_time = time.time() - epoch_start

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    print(f"  Epoch Time: {epoch_time/60:.1f} minutes")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Update learning rate scheduler
    scheduler.step(val_loss)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_binary_hornet_model.pth')
        print(f"  *** New best validation accuracy! Model saved. ***")

    # Estimate remaining time
    elapsed = time.time() - training_start
    avg_epoch_time = elapsed / (epoch + 1)
    remaining = avg_epoch_time * (EPOCHS - epoch - 1)
    print(f"  Estimated remaining time: {remaining/60:.1f} minutes ({remaining/3600:.1f} hours)")

total_time = time.time() - training_start
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

# ============== FINAL EVALUATION ==============
print("\n" + "="*70)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*70)

model.load_state_dict(torch.load('best_binary_hornet_model.pth'))
_, final_acc, final_preds, final_labels = validate(model, val_loader, criterion, DEVICE)

print(f"\nFinal Validation Accuracy: {final_acc:.2f}%")

# Confusion Matrix
cm = confusion_matrix(final_labels, final_preds)
print("\nConfusion Matrix:")
print("                  Predicted")
print("                  Negative  Positive")
print(f"Actual Negative     {cm[0,0]:6d}    {cm[0,1]:6d}")
print(f"Actual Positive     {cm[1,0]:6d}    {cm[1,1]:6d}")

# Classification Report
print("\nClassification Report:")
print(classification_report(final_labels, final_preds,
                          target_names=['Not Asian Hornet', 'Asian Hornet'],
                          digits=4))

print("\n" + "="*70)
print("Model saved as: best_binary_hornet_model.pth")
print("="*70)
