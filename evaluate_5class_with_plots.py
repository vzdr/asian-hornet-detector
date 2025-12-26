"""
Evaluate 5-Class Model with Professional Plots
Uses supervisor's plotting functions to generate ROC-AUC, PR curves, and confusion matrices

Run with: py -3.10 evaluate_5class_with_plots.py
"""

import sys
import os

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_curve
from matplotlib import pyplot as plt
import pickle

# Configuration
MODEL_PATH = "multiclass_models/best_5class_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
NUM_CLASSES = 5
CLASS_NAMES = ['Asian Hornet', 'Bee', 'European Hornet', 'Wasp', 'Random Images']
BATCH_SIZE = 64

# Output directory
OUTPUT_DIR = Path("5class_evaluation_plots")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_cifar10_batch(file_path):
    """Load a CIFAR-10 batch file (Python pickle format)"""
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')

    images = data_dict[b'data']
    labels = data_dict[b'labels']

    # Reshape from (N, 3072) to (N, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    pil_images = []
    for img in images:
        pil_img = Image.fromarray(img)
        pil_images.append(pil_img)

    return pil_images, labels


class FiveClassDataset(Dataset):
    """Dataset for 5-class classification including random images"""

    def __init__(self, data_sources, split='valid', transform=None):
        self.transform = transform
        self.samples = []  # Will store (image_or_path, label, is_pil)

        # Class mapping
        self.class_to_idx = {
            'asian_hornets': 0,
            'bees': 1,
            'european_hornets': 2,
            'wasps': 3,
            'random_images': 4
        }

        print(f"\nLoading {split.upper()} data...")

        # Load from YOLO datasets (Asian hornets and bees)
        for dataset_name in ['bees_hornets1', 'bees_hornets2']:
            if dataset_name not in data_sources:
                continue

            root = Path(data_sources[dataset_name])
            labels_dir = root / split / 'labels'
            images_dir = root / split / 'images'

            if not labels_dir.exists() or not images_dir.exists():
                continue

            for label_file in labels_dir.glob('*.txt'):
                try:
                    with open(label_file) as f:
                        line = f.readline().strip()
                        if not line:
                            continue

                        # YOLO: class 0 = bee, class 1 = asian hornet
                        yolo_class = int(line.split()[0])

                        # Find image
                        img_path = images_dir / f"{label_file.stem}.jpg"
                        if not img_path.exists():
                            img_path = images_dir / f"{label_file.stem}.jpeg"

                        if img_path.exists():
                            if yolo_class == 0:  # bee
                                self.samples.append((str(img_path), self.class_to_idx['bees'], False))
                            elif yolo_class == 1:  # asian hornet
                                self.samples.append((str(img_path), self.class_to_idx['asian_hornets'], False))
                except:
                    continue

        # Load European hornets from GBIF
        if 'gbif_european_hornets' in data_sources:
            euro_dir = Path(data_sources['gbif_european_hornets'])
            if euro_dir.exists():
                for img_path in euro_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx['european_hornets'], False))
                for img_path in euro_dir.glob('*.jpeg'):
                    self.samples.append((str(img_path), self.class_to_idx['european_hornets'], False))

        # Load wasps from GBIF
        if 'gbif_wasps' in data_sources:
            wasp_dir = Path(data_sources['gbif_wasps'])
            if wasp_dir.exists():
                for img_path in wasp_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx['wasps'], False))
                for img_path in wasp_dir.glob('*.jpeg'):
                    self.samples.append((str(img_path), self.class_to_idx['wasps'], False))

        # Load random images
        self._load_random_images(data_sources, split)

        print(f"Loaded {len(self.samples)} total samples")
        # Print class distribution
        class_counts = [0] * 5
        for _, label, _ in self.samples:
            class_counts[label] += 1
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {class_counts[i]}")

    def _load_random_images(self, data_sources, split):
        """Load random images from CIFAR-10 and other sources"""
        label = self.class_to_idx['random_images']

        # 1. Load from CIFAR-10
        if 'cifar10' in data_sources:
            cifar_dir = Path(data_sources['cifar10'])
            if cifar_dir.exists():
                if split == 'train':
                    # Load training batches 1-5
                    for i in range(1, 6):
                        batch_file = cifar_dir / f'data_batch_{i}'
                        if batch_file.exists():
                            pil_images, _ = load_cifar10_batch(str(batch_file))
                            for pil_img in pil_images:
                                self.samples.append((pil_img, label, True))
                else:  # valid
                    # Load test batch
                    test_batch = cifar_dir / 'test_batch'
                    if test_batch.exists():
                        pil_images, _ = load_cifar10_batch(str(test_batch))
                        for pil_img in pil_images:
                            self.samples.append((pil_img, label, True))

        # 2. Load from randomimagesdataset
        if 'randomimagesdataset' in data_sources:
            random_dir = Path(data_sources['randomimagesdataset'])
            split_dir = random_dir / split
            if split_dir.exists():
                for img_path in split_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), label, False))
                for img_path in split_dir.glob('*.jpeg'):
                    self.samples.append((str(img_path), label, False))
                for img_path in split_dir.glob('*.png'):
                    self.samples.append((str(img_path), label, False))

        # 3. Load from randomimages (only for training)
        if split == 'train' and 'randomimages' in data_sources:
            random_dir = Path(data_sources['randomimages'])
            if random_dir.exists():
                for img_path in random_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), label, False))
                for img_path in random_dir.glob('*.jpeg'):
                    self.samples.append((str(img_path), label, False))
                for img_path in random_dir.glob('*.png'):
                    self.samples.append((str(img_path), label, False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_or_path, label, is_pil = self.samples[idx]

        try:
            if is_pil:
                # Already a PIL image (from CIFAR-10)
                image = img_or_path.convert('RGB')
            else:
                # Load from file path
                image = Image.open(img_or_path).convert('RGB')
        except:
            # Fallback for corrupted images
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def load_model():
    """Load trained 5-class model"""
    print(f"Loading model from {MODEL_PATH}...")
    model = models.efficientnet_b3(pretrained=False)

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    return model


def evaluate_model(model, dataloader):
    """Run inference and collect predictions"""
    print("\nRunning inference on validation set...")

    all_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_pred_probs = np.array(all_pred_probs)

    print(f"Collected {len(all_labels)} predictions")
    print(f"Probability shape: {all_pred_probs.shape}")

    return all_labels, all_pred_probs


# ============================================================================
# SUPERVISOR'S PLOTTING FUNCTIONS (adapted from plot_prauc_auc_confusion_matrix.py)
# ============================================================================

def to_categorical(y, num_classes=None, dtype='float32'):
    """Convert class labels to one-hot encoded format"""
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def plot_ROC_AUC_multiple_class(classes, y_true, y_pred_prob, plot_dir=None, filename=None, show=False,
                                plot_and_save=False, replot=False):
    """Plot ROC-AUC curves for multi-class classification"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    threshold = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], threshold[i] = sklearn.metrics.roc_curve(to_categorical(y_true)[:, i], y_pred_prob[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve
    fpr["micro"], tpr["micro"], threshold["micro"] = sklearn.metrics.roc_curve(
        to_categorical(y_true).ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

    if plot_and_save:
        os.makedirs(plot_dir, exist_ok=True)
        with open(os.path.join(plot_dir, filename + '_roc_micro_' + '%.4f' % roc_auc["micro"] + '.txt'), "w") as text_file:
            print(roc_auc, file=text_file)

    lw = 2
    # Compute macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    # Plot
    if plot_and_save:
        png_file = os.path.join(plot_dir, filename + '_ROC_AUC_' + '%.4f' % roc_auc["micro"] + '.pdf')

        if not os.path.exists(png_file) or replot:
            plt.clf()
            plt.close()
            plt.figure(figsize=(8, 8))
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.3f})'.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.3f})'.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            for i in range(len(classes)):
                plt.plot(fpr[i], tpr[i], lw=lw,
                         label='{0} (area = {1:0.3f})'.format(classes[i], roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title('ROC AUC micro: %.4f , macro: %.4f ' % (roc_auc["micro"], roc_auc["macro"]))
            plt.xlabel('False positives rate(FPR) / (1-Specificity)')
            plt.ylabel('True positives rate(TPR) / Sensitivity / Recall')
            plt.legend(loc="lower right")

            plt.tight_layout()
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.savefig(png_file)
            if show:
                plt.show()
            plt.close()
            plt.clf()
    return roc_auc


def plot_PR_curve_AUC_multiple_class(classes, y_true, y_pred_prob, plot_dir, filename, show=False, replot=False):
    """Plot Precision-Recall curves for multi-class classification"""
    precision = dict()
    recall = dict()
    threshold = dict()
    average_precision = dict()
    Y_test = to_categorical(y_true)
    y_score = y_pred_prob

    n_classes = len(classes)

    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # Micro-average
    precision["micro"], recall["micro"], threshold["micro"] = precision_recall_curve(
        Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(plot_dir, filename + '_pr_' + '%.4f' % average_precision["micro"] + '.txt'), "w") as text_file:
        print(average_precision, file=text_file)

    png_file = os.path.join(plot_dir, filename + '_MSC_PR_' + '%.4f' % average_precision["micro"] + '.pdf')
    if replot or not os.path.exists(png_file):
        plt.clf()
        plt.close()

        plt.figure(figsize=(8, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average (area = {0:0.3f})'.format(average_precision["micro"]))

        for i in range(n_classes):
            l, = plt.plot(recall[i], precision[i], lw=2)
            lines.append(l)
            labels.append('{0} (area = {1:0.3f})'.format(classes[i], average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average Precision / PR_Curve_AUC micro: %.4f' % (average_precision["micro"]))
        plt.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig(png_file, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()
        plt.clf()
    return average_precision


def plot_confusion_matrix(cm, classes, filename=None, normalize=False, cmap=plt.cm.Blues, figsize=(8, 8), replot=False):
    """Plot confusion matrix"""
    if normalize:
        cm_list = [cm]
        cm = []
        for item in cm_list:
            cm.append(item.astype('float') / item.sum(axis=1)[:, np.newaxis] * 100)
        cm = np.mean(cm, axis=0)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    if not os.path.exists(filename) or replot:
        plt.clf()
        plt.close()
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = '.2f' if normalize else '.2f'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(filename)
        plt.clf()
        plt.close()
    return cm


def main():
    print("="*70)
    print("5-CLASS MODEL EVALUATION WITH PROFESSIONAL PLOTS")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load model
    model = load_model()

    # Prepare data
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

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = FiveClassDataset(data_sources, split='valid', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Run evaluation
    y_true, y_pred_probs = evaluate_model(model, val_loader)

    # Get hard predictions
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = np.mean(y_pred[class_mask] == i) * 100
            print(f"  {class_name}: {class_acc:.2f}% ({class_mask.sum()} samples)")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PROFESSIONAL PLOTS")
    print("="*70)

    # 1. ROC-AUC Curves
    print("\n1. Generating ROC-AUC curves...")
    roc_auc = plot_ROC_AUC_multiple_class(
        classes=CLASS_NAMES,
        y_true=y_true,
        y_pred_prob=y_pred_probs,
        plot_dir=str(OUTPUT_DIR),
        filename='5class',
        show=False,
        plot_and_save=True,
        replot=True
    )
    print(f"   ROC-AUC scores:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"     {class_name}: {roc_auc[i]:.4f}")
    print(f"     Micro-average: {roc_auc['micro']:.4f}")
    print(f"     Macro-average: {roc_auc['macro']:.4f}")

    # 2. Precision-Recall Curves
    print("\n2. Generating Precision-Recall curves...")
    pr_auc = plot_PR_curve_AUC_multiple_class(
        classes=CLASS_NAMES,
        y_true=y_true,
        y_pred_prob=y_pred_probs,
        plot_dir=str(OUTPUT_DIR),
        filename='5class',
        show=False,
        replot=True
    )
    print(f"   Average Precision scores:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"     {class_name}: {pr_auc[i]:.4f}")
    print(f"     Micro-average: {pr_auc['micro']:.4f}")

    # 3. Confusion Matrix
    print("\n3. Generating confusion matrices...")
    cm = confusion_matrix(y_true, y_pred)

    # Raw counts
    cm_file = str(OUTPUT_DIR / '5class_cm.png')
    plot_confusion_matrix(
        cm,
        classes=CLASS_NAMES,
        filename=cm_file,
        normalize=False,
        figsize=(12, 12),
        replot=True
    )
    print(f"   Raw confusion matrix saved: {cm_file}")

    # Normalized percentages
    cm_norm_file = str(OUTPUT_DIR / '5class_cm_normalize.png')
    plot_confusion_matrix(
        cm,
        classes=CLASS_NAMES,
        filename=cm_norm_file,
        normalize=True,
        figsize=(12, 12),
        replot=True
    )
    print(f"   Normalized confusion matrix saved: {cm_norm_file}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print(f"  - ROC-AUC plot (PDF)")
    print(f"  - Precision-Recall plot (PDF)")
    print(f"  - Confusion matrix - raw counts (PNG)")
    print(f"  - Confusion matrix - normalized % (PNG)")
    print(f"  - Metric text files")
    print("="*70)


if __name__ == '__main__':
    main()
