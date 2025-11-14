import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
IMG_SIZE = 224
NUM_WORKERS = 0
MODEL_PATH = "best_binary_hornet_model.pth"

# Paths
YOLO1_IMAGES_DIR = Path("D:/Ultimate Dataset/BeesAndHornets1/Bee And Asian Hornet Detection/train/images")
YOLO1_LABELS_DIR = Path("D:/Ultimate Dataset/BeesAndHornets1/Bee And Asian Hornet Detection/train/labels")
YOLO2_IMAGES_DIR = Path("D:/Ultimate Dataset/BeesAndHornets2/Dataset/train/images")
YOLO2_LABELS_DIR = Path("D:/Ultimate Dataset/BeesAndHornets2/Dataset/train/labels")
EUROPEAN_HORNET_DIR = Path("D:/Ultimate Dataset/european_hornets_gbif")
WASP_DIR = Path("D:/Ultimate Dataset/wasps_gbif")

class HornetDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def plot_confusion_matrix(cm, class_names, title, save_path, metrics):
    """Plot and save confusion matrix with metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title}\nConfusion Matrix', fontsize=14, fontweight='bold')

    # Plot normalized confusion matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage (%)'}, annot_kws={'size': 14})
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title}\nNormalized Confusion Matrix (%)', fontsize=14, fontweight='bold')

    # Add metrics as text below the plots
    metrics_text = f"""
    Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)

    Not Asian Hornet:
        Precision: {metrics['not_hornet']['precision']:.4f}
        Recall: {metrics['not_hornet']['recall']:.4f}
        F1-Score: {metrics['not_hornet']['f1']:.4f}

    Asian Hornet:
        Precision: {metrics['hornet']['precision']:.4f}
        Recall: {metrics['hornet']['recall']:.4f}
        F1-Score: {metrics['hornet']['f1']:.4f}

    Macro Avg F1: {metrics['macro_f1']:.4f}
    Weighted Avg F1: {metrics['weighted_f1']:.4f}
    """

    fig.text(0.5, -0.15, metrics_text, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()

def load_model():
    """Load the trained model"""
    print("Loading model...")
    model = models.efficientnet_b3(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    return model

def evaluate_dataset(model, dataloader, dataset_name):
    """Evaluate model on a dataset"""
    print(f"\nEvaluating on {dataset_name}...")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )

    # Macro and weighted averages
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )

    metrics = {
        'accuracy': accuracy,
        'not_hornet': {
            'precision': precision[0],
            'recall': recall[0],
            'f1': f1[0],
            'support': support[0]
        },
        'hornet': {
            'precision': precision[1],
            'recall': recall[1],
            'f1': f1[1],
            'support': support[1]
        },
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

    return cm, metrics, all_preds, all_labels

def load_validation_set():
    """Load the same validation set used during training (10% split with seed 42)"""
    print("\n" + "="*70)
    print("LOADING VALIDATION SET (6,559 images)")
    print("="*70)

    # Load all Asian hornets
    asian_hornet_images = []
    bee_images = []

    # Parse YOLO1
    for label_file in YOLO1_LABELS_DIR.glob("*.txt"):
        img_name = label_file.stem + ".jpg"
        img_path = YOLO1_IMAGES_DIR / img_name
        if img_path.exists():
            with open(label_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    if class_id == 1:
                        asian_hornet_images.append(str(img_path))
                    elif class_id == 0:
                        bee_images.append(str(img_path))

    # Parse YOLO2
    for label_file in YOLO2_LABELS_DIR.glob("*.txt"):
        img_name = label_file.stem + ".jpg"
        img_path = YOLO2_IMAGES_DIR / img_name
        if img_path.exists():
            with open(label_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    if class_id == 1:
                        asian_hornet_images.append(str(img_path))
                    elif class_id == 0:
                        bee_images.append(str(img_path))

    # Load European hornets
    european_hornet_images = [str(p) for p in EUROPEAN_HORNET_DIR.glob("*.jpg")]

    # Load wasps
    wasp_images = [str(p) for p in WASP_DIR.glob("*.jpg")]

    print(f"Loaded Asian Hornets: {len(asian_hornet_images)}")
    print(f"Loaded Bees: {len(bee_images)}")
    print(f"Loaded European Hornets: {len(european_hornet_images)}")
    print(f"Loaded Wasps: {len(wasp_images)}")

    # Create balanced dataset (same as training)
    all_samples = []

    # Use ALL Asian hornets
    for img_path in asian_hornet_images:
        all_samples.append((img_path, 1))

    # Combine negative class
    negative_images = bee_images + european_hornet_images + wasp_images
    np.random.seed(42)
    np.random.shuffle(negative_images)
    negative_samples = negative_images[:len(asian_hornet_images)]

    for img_path in negative_samples:
        all_samples.append((img_path, 0))

    # Shuffle all samples with same seed as training
    np.random.seed(42)
    np.random.shuffle(all_samples)

    # Split: 90% train, 10% validation (same as training)
    train_size = int(0.9 * len(all_samples))
    val_samples = all_samples[train_size:]

    val_paths = [s[0] for s in val_samples]
    val_labels = [s[1] for s in val_samples]

    print(f"\nValidation set: {len(val_samples)} images")
    print(f"  Positive (Asian Hornets): {sum(val_labels)}")
    print(f"  Negative (Others): {len(val_labels) - sum(val_labels)}")

    return val_paths, val_labels

def load_test_set():
    """Load unused negative images as test set"""
    print("\n" + "="*70)
    print("LOADING TEST SET (Unused ~78k images)")
    print("="*70)

    # Load all Asian hornets
    asian_hornet_images = []
    bee_images = []

    # Parse YOLO1
    for label_file in YOLO1_LABELS_DIR.glob("*.txt"):
        img_name = label_file.stem + ".jpg"
        img_path = YOLO1_IMAGES_DIR / img_name
        if img_path.exists():
            with open(label_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    if class_id == 1:
                        asian_hornet_images.append(str(img_path))
                    elif class_id == 0:
                        bee_images.append(str(img_path))

    # Parse YOLO2
    for label_file in YOLO2_LABELS_DIR.glob("*.txt"):
        img_name = label_file.stem + ".jpg"
        img_path = YOLO2_IMAGES_DIR / img_name
        if img_path.exists():
            with open(label_file) as f:
                first_line = f.readline().strip()
                if first_line:
                    class_id = int(first_line.split()[0])
                    if class_id == 1:
                        asian_hornet_images.append(str(img_path))
                    elif class_id == 0:
                        bee_images.append(str(img_path))

    # Load European hornets
    european_hornet_images = [str(p) for p in EUROPEAN_HORNET_DIR.glob("*.jpg")]

    # Load wasps
    wasp_images = [str(p) for p in WASP_DIR.glob("*.jpg")]

    # Get the negative images that were used in training/validation
    negative_images_all = bee_images + european_hornet_images + wasp_images

    # Recreate the same selection as training to identify unused images
    np.random.seed(42)
    indices = np.arange(len(negative_images_all))
    np.random.shuffle(indices)
    used_indices = indices[:len(asian_hornet_images)]
    unused_indices = indices[len(asian_hornet_images):]

    # Get unused negative images
    unused_negative_images = [negative_images_all[i] for i in unused_indices]

    test_paths = unused_negative_images
    test_labels = [0] * len(test_paths)  # All negative class

    print(f"\nTest set: {len(test_paths)} images")
    print(f"  All Negative class (Bees, European Hornets, Wasps)")
    print(f"  This tests the model's specificity to Asian hornets")

    return test_paths, test_labels

def main():
    print("="*70)
    print("BINARY ASIAN HORNET CLASSIFIER - EVALUATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    # Load model
    model = load_model()

    # Define transforms (no augmentation for evaluation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ============================================
    # 1. EVALUATE ON VALIDATION SET (6k images)
    # ============================================
    val_paths, val_labels = load_validation_set()
    val_dataset = HornetDataset(val_paths, val_labels, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    val_cm, val_metrics, val_preds, val_true = evaluate_dataset(model, val_loader, "Validation Set (6k)")

    # Print detailed report
    print("\n" + "="*70)
    print("VALIDATION SET RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Negative  Positive")
    print(f"Actual Negative   {val_cm[0][0]:8d}  {val_cm[0][1]:8d}")
    print(f"Actual Positive   {val_cm[1][0]:8d}  {val_cm[1][1]:8d}")
    print(f"\nClass: Not Asian Hornet")
    print(f"  Precision: {val_metrics['not_hornet']['precision']:.4f}")
    print(f"  Recall:    {val_metrics['not_hornet']['recall']:.4f}")
    print(f"  F1-Score:  {val_metrics['not_hornet']['f1']:.4f}")
    print(f"  Support:   {val_metrics['not_hornet']['support']}")
    print(f"\nClass: Asian Hornet")
    print(f"  Precision: {val_metrics['hornet']['precision']:.4f}")
    print(f"  Recall:    {val_metrics['hornet']['recall']:.4f}")
    print(f"  F1-Score:  {val_metrics['hornet']['f1']:.4f}")
    print(f"  Support:   {val_metrics['hornet']['support']}")
    print(f"\nMacro Avg F1:    {val_metrics['macro_f1']:.4f}")
    print(f"Weighted Avg F1: {val_metrics['weighted_f1']:.4f}")

    # Generate visualization
    plot_confusion_matrix(
        val_cm,
        class_names=['Not Asian Hornet', 'Asian Hornet'],
        title='Validation Set (6,559 images)',
        save_path='confusion_matrix_validation.png',
        metrics=val_metrics
    )

    # ============================================
    # 2. EVALUATE ON TEST SET (~78k unused images)
    # ============================================
    test_paths, test_labels = load_test_set()
    test_dataset = HornetDataset(test_paths, test_labels, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    test_cm, test_metrics, test_preds, test_true = evaluate_dataset(model, test_loader, "Test Set (~78k)")

    # Print detailed report
    print("\n" + "="*70)
    print("TEST SET RESULTS (Unused Negative Images)")
    print("="*70)
    print(f"\nOverall Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Negative  Positive")
    print(f"Actual Negative   {test_cm[0][0]:8d}  {test_cm[0][1]:8d}")
    if test_cm.shape[0] > 1:
        print(f"Actual Positive   {test_cm[1][0]:8d}  {test_cm[1][1]:8d}")
    print(f"\nClass: Not Asian Hornet")
    print(f"  Precision: {test_metrics['not_hornet']['precision']:.4f}")
    print(f"  Recall:    {test_metrics['not_hornet']['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['not_hornet']['f1']:.4f}")
    print(f"  Support:   {test_metrics['not_hornet']['support']}")

    if test_cm.shape[0] > 1:
        print(f"\nClass: Asian Hornet")
        print(f"  Precision: {test_metrics['hornet']['precision']:.4f}")
        print(f"  Recall:    {test_metrics['hornet']['recall']:.4f}")
        print(f"  F1-Score:  {test_metrics['hornet']['f1']:.4f}")
        print(f"  Support:   {test_metrics['hornet']['support']}")

    print(f"\nMacro Avg F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted Avg F1: {test_metrics['weighted_f1']:.4f}")

    # Calculate false positive rate
    false_positives = test_cm[0][1] if test_cm.shape[1] > 1 else 0
    total_negatives = test_cm[0][0] + false_positives
    fpr = false_positives / total_negatives if total_negatives > 0 else 0
    print(f"\nFalse Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  (Out of {total_negatives} negative images, {false_positives} were incorrectly classified as Asian hornets)")

    # Generate visualization
    plot_confusion_matrix(
        test_cm,
        class_names=['Not Asian Hornet', 'Asian Hornet'],
        title=f'Test Set ({len(test_paths):,} unused images)',
        save_path='confusion_matrix_test.png',
        metrics=test_metrics
    )

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("Generated files:")
    print("  - confusion_matrix_validation.png")
    print("  - confusion_matrix_test.png")

if __name__ == "__main__":
    main()
