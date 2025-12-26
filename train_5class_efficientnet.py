"""
Multi-Class Asian Hornet Classifier Training Script (5 Classes)
================================================================

5-class classification:
- Class 0: Asian Hornets (~45k images)
- Class 1: Bees (~45k images)
- Class 2: European Hornets (~38k images)
- Class 3: Wasps (~31k images)
- Class 4: Random Images (CIFAR-10 + other random datasets)

This script trains an EfficientNet-B3 model with:
- CrossEntropyLoss (multi-class classification)
- Softmax activation for natural confidence calibration
- Balanced sampling (30k per class)
- Data augmentation for generalization

Key Improvement over 4-Class:
- Learns to reject non-insect images (random class)
- More robust against false positives in real-world deployment
"""

import os
import sys
import torch
import pickle

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


def load_cifar10_batch(file_path):
    """Load a CIFAR-10 batch file (Python pickle format)"""
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')

    # Extract images and labels
    images = data_dict[b'data']
    labels = data_dict[b'labels']

    # Reshape images from (N, 3072) to (N, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Convert to list of PIL Images
    pil_images = []
    for img in images:
        pil_img = Image.fromarray(img)
        pil_images.append(pil_img)

    return pil_images, labels


class FiveClassDataset(Dataset):
    """
    Dataset for 5-class hornet/bee/wasp/random classification
    Loads from YOLO datasets (Asian hornets + bees), GBIF (European hornets + wasps),
    and random images (CIFAR-10 + other sources)
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
        self.samples = []  # List of (img_or_path, label, class_name, is_pil)

        # Class mapping
        self.class_to_idx = {
            'asian_hornets': 0,
            'bees': 1,
            'european_hornets': 2,
            'wasps': 3,
            'random_images': 4
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.class_names = ['Asian Hornet', 'Bee', 'European Hornet', 'Wasp', 'Random Images']

        print(f"\nLoading {split.upper()} data...")

        # Collect all images by class
        class_images = {class_name: [] for class_name in self.class_to_idx.keys()}

        # 1. Load Asian hornets and Bees from YOLO datasets
        self._load_yolo_data(data_sources, split, class_images)

        # 2. Load European hornets from GBIF
        self._load_gbif_european_hornets(data_sources, class_images)

        # 3. Load Wasps from GBIF
        self._load_gbif_wasps(data_sources, class_images)

        # 4. Load Random Images (CIFAR-10 + others)
        self._load_random_images(data_sources, split, class_images)

        # Print raw counts
        print("\nRaw image counts:")
        for class_name, images in class_images.items():
            if not class_name.startswith('_'):  # Skip metadata keys
                print(f"  {class_name}: {len(images)}")

        # Sample if requested (for balanced training)
        if samples_per_class and split == 'train':
            print(f"\nBalancing dataset: {samples_per_class} images per class...")
            for class_name, images in class_images.items():
                # Skip metadata keys
                if class_name.startswith('_'):
                    continue

                # Special handling for random_images: enforce 70/30 garden/generic split
                if class_name == 'random_images' and '_garden_count' in class_images:
                    garden_count = class_images['_garden_count']
                    generic_count = class_images['_generic_count']

                    # Calculate 70/30 split
                    garden_needed = int(samples_per_class * 0.7)
                    generic_needed = samples_per_class - garden_needed

                    print(f"\n  Random images 70/30 split:")
                    print(f"    Garden images needed: {garden_needed} (available: {garden_count})")
                    print(f"    Generic images needed: {generic_needed} (available: {generic_count})")

                    # Sample from garden pool (first N images are garden)
                    garden_pool = images[:garden_count]
                    generic_pool = images[garden_count:]

                    sampled_garden = random.sample(garden_pool, min(garden_needed, len(garden_pool)))
                    sampled_generic = random.sample(generic_pool, min(generic_needed, len(generic_pool)))

                    sampled = sampled_garden + sampled_generic
                    print(f"    Final mix: {len(sampled_garden)} garden + {len(sampled_generic)} generic = {len(sampled)} total")
                elif len(images) > samples_per_class:
                    sampled = random.sample(images, samples_per_class)
                else:
                    sampled = images
                    print(f"  Warning: {class_name} has only {len(images)} images (< {samples_per_class})")

                label = self.class_to_idx[class_name]
                for item in sampled:
                    # item can be either a file path string or (PIL_image, is_pil_flag) tuple
                    if isinstance(item, tuple):
                        pil_img, is_pil = item
                        self.samples.append((pil_img, label, class_name, True))
                    else:
                        self.samples.append((item, label, class_name, False))
        else:
            # Use all available images
            for class_name, images in class_images.items():
                # Skip metadata keys
                if class_name.startswith('_'):
                    continue

                label = self.class_to_idx[class_name]
                for item in images:
                    if isinstance(item, tuple):
                        pil_img, is_pil = item
                        self.samples.append((pil_img, label, class_name, True))
                    else:
                        self.samples.append((item, label, class_name, False))

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

    def _load_random_images(self, data_sources, split, class_images):
        """Load random images: 70% garden/nature (FlowerBackgrounds) + 30% generic random (CIFAR-10, etc.)"""
        garden_images = []
        generic_random_images = []

        # 1. Load from FlowerBackgrounds (real flower/garden photos - 70% of final mix)
        if 'flower_backgrounds' in data_sources:
            flowers_dir = Path(data_sources['flower_backgrounds'])
            subdir = 'train' if split == 'train' else 'test'
            images_dir = flowers_dir / subdir

            if images_dir.exists():
                print(f"  Loading real flower garden images from {subdir}...")
                # Load from both subdirectories (flowers_recognition and flowers299)
                for subdir_name in ['flowers_recognition', 'flowers299']:
                    subdir_path = images_dir / subdir_name
                    if subdir_path.exists():
                        for img_path in subdir_path.rglob('*.jpg'):
                            garden_images.append(str(img_path))
                        for img_path in subdir_path.rglob('*.jpeg'):
                            garden_images.append(str(img_path))
                        for img_path in subdir_path.rglob('*.png'):
                            garden_images.append(str(img_path))
                print(f"  Loaded {len(garden_images)} real flower images from FlowerBackgrounds/{subdir}")
            else:
                print(f"  Warning: FlowerBackgrounds {subdir} directory not found: {images_dir}")

        # 2. Load from CIFAR-10 (generic random - will be 30% of final mix)
        if 'cifar10' in data_sources:
            cifar_dir = Path(data_sources['cifar10'])

            if split == 'train':
                # Load training batches (data_batch_1 to data_batch_5)
                print("  Loading CIFAR-10 training batches...")
                for i in range(1, 6):
                    batch_file = cifar_dir / f'data_batch_{i}'
                    if batch_file.exists():
                        images, _ = load_cifar10_batch(batch_file)
                        # Store as (PIL_image, is_pil=True) tuples
                        generic_random_images.extend([(img, True) for img in images])
                print(f"  Loaded {len(generic_random_images)} images from CIFAR-10 train")
            else:
                # Load test batch
                test_batch = cifar_dir / 'test_batch'
                if test_batch.exists():
                    print("  Loading CIFAR-10 test batch...")
                    images, _ = load_cifar10_batch(test_batch)
                    generic_random_images.extend([(img, True) for img in images])
                    print(f"  Loaded {len(generic_random_images)} images from CIFAR-10 test")

        # 3. Load from randomimagesdataset
        if 'randomimagesdataset' in data_sources:
            dataset_dir = Path(data_sources['randomimagesdataset'])
            subdir = 'train' if split == 'train' else 'test'
            images_dir = dataset_dir / subdir

            if images_dir.exists():
                for img_path in images_dir.glob('*.jpg'):
                    generic_random_images.append(str(img_path))
                for img_path in images_dir.glob('*.jpeg'):
                    generic_random_images.append(str(img_path))
                for img_path in images_dir.glob('*.png'):
                    generic_random_images.append(str(img_path))
                count_added = len([x for x in generic_random_images if isinstance(x, str) and 'randomimagesdataset' in str(x)])
                print(f"  Loaded images from randomimagesdataset/{subdir}")

        # 4. Load from randomimages (only for training, no test split)
        if split == 'train' and 'randomimages' in data_sources:
            random_dir = Path(data_sources['randomimages'])
            if random_dir.exists():
                count_before = len(generic_random_images)
                for img_path in random_dir.glob('*.jpg'):
                    generic_random_images.append(str(img_path))
                for img_path in random_dir.glob('*.jpeg'):
                    generic_random_images.append(str(img_path))
                for img_path in random_dir.glob('*.png'):
                    generic_random_images.append(str(img_path))
                count_after = len(generic_random_images)
                print(f"  Loaded {count_after - count_before} images from randomimages")

        # 5. Mix 70% garden/nature + 30% generic random
        print(f"\n  Mixing random images: 70% garden/nature + 30% generic random")
        print(f"    Available garden images: {len(garden_images)}")
        print(f"    Available generic images: {len(generic_random_images)}")

        # Shuffle both pools
        import random as rand_module
        rand_module.shuffle(garden_images)
        rand_module.shuffle(generic_random_images)

        # Calculate 70/30 split based on samples_per_class
        # We'll handle the final mixing in the sampling stage
        # For now, just combine them all and let the sampling do the 70/30 split
        combined_random = garden_images + generic_random_images

        # Store with metadata to allow 70/30 sampling later
        class_images['random_images'] = combined_random
        class_images['_garden_count'] = len(garden_images)
        class_images['_generic_count'] = len(generic_random_images)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_or_path, label, class_name, is_pil = self.samples[idx]

        # Load image
        try:
            if is_pil:
                # Already a PIL image (from CIFAR-10)
                image = img_or_path.convert('RGB') if img_or_path.mode != 'RGB' else img_or_path
            else:
                # Load from file path
                image = Image.open(img_or_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
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
    print("5-CLASS ASIAN HORNET CLASSIFIER TRAINING")
    print("="*70)
    print("\n5-Class Classification:")
    print("  0: Asian Hornets")
    print("  1: Bees")
    print("  2: European Hornets")
    print("  3: Wasps")
    print("  4: Random Images (non-insects)")

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
        'gbif_wasps': r'D:\Ultimate Dataset\wasps_gbif',
        'flower_backgrounds': r'D:\Ultimate Dataset\FlowerBackgrounds',  # Real flower/garden photos for realistic backgrounds
        'cifar10': r'D:\Ultimate Dataset\cifar-10-python\cifar-10-batches-py',
        'randomimagesdataset': r'D:\Ultimate Dataset\randomimagesdataset\dataset',
        'randomimages': r'D:\Ultimate Dataset\randomimages\data'
    }

    # Hyperparameters
    BATCH_SIZE = 64  # Increase if GPU memory allows
    LEARNING_RATE = 0.0001  # Lower for fine-tuning
    EPOCHS = 15
    NUM_WORKERS = 4  # Parallel data loading for speed
    SAMPLES_PER_CLASS = 30000  # 30k per class = 150k total

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

    train_dataset = FiveClassDataset(
        data_sources=data_sources,
        split='train',
        transform=train_transform,
        samples_per_class=SAMPLES_PER_CLASS
    )

    val_dataset = FiveClassDataset(
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

    # Modify classifier for 5 classes
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 5)  # Changed from 4 to 5
    )

    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model: EfficientNet-B3 (5 classes)")
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
            torch.save(model.state_dict(), output_dir / 'best_5class_model.pth')
            print(f"  *** New best model saved! Val Acc: {val_acc:.2f}% ***")

    # Training time
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time/60:.1f} minutes")

    # Save training history
    with open(output_dir / '5class_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    model.load_state_dict(torch.load(output_dir / 'best_5class_model.pth'))
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

    print(f"\nBest Validation Accuracy: {val_acc:.2f}%")

    # Classification report
    class_names = ['Asian Hornet', 'Bee', 'European Hornet', 'Wasp', 'Random Images']
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

    with open(output_dir / '5class_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model saved to: {output_dir / 'best_5class_model.pth'}")
    print(f"Results saved to: {output_dir / '5class_results.json'}")


if __name__ == '__main__':
    main()
