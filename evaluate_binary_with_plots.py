"""
Evaluate Binary EfficientNet Model with Professional Plots
Uses supervisor's plotting functions to generate ROC-AUC, PR curves, and confusion matrices
Binary classification: Asian Hornet vs NOT Asian Hornet

Run with: py -3.10 evaluate_binary_with_plots.py
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

# Configuration
MODEL_PATH = "image_classifier/models/best_binary_efficientnet.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
NUM_CLASSES = 2  # Binary classification
CLASS_NAMES = ['NOT Asian Hornet', 'Asian Hornet']
BATCH_SIZE = 64
TEMPERATURE = 4.0  # Temperature scaling to prevent sigmoid saturation

# Output directory
OUTPUT_DIR = Path("binary_evaluation_plots")
OUTPUT_DIR.mkdir(exist_ok=True)


class BinaryDataset(Dataset):
    """Dataset for binary classification: Asian Hornet vs Everything Else"""

    def __init__(self, data_sources, split='valid', transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

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
                            if yolo_class == 0:  # bee -> NOT Asian Hornet
                                self.image_paths.append(str(img_path))
                                self.labels.append(0)
                            elif yolo_class == 1:  # asian hornet -> Asian Hornet
                                self.image_paths.append(str(img_path))
                                self.labels.append(1)
                except:
                    continue

        # Load European hornets from GBIF -> NOT Asian Hornet
        if 'gbif_european_hornets' in data_sources:
            euro_dir = Path(data_sources['gbif_european_hornets'])
            if euro_dir.exists():
                for img_path in euro_dir.glob('*.jpg'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(0)  # NOT Asian Hornet
                for img_path in euro_dir.glob('*.jpeg'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(0)  # NOT Asian Hornet

        # Load wasps from GBIF -> NOT Asian Hornet
        if 'gbif_wasps' in data_sources:
            wasp_dir = Path(data_sources['gbif_wasps'])
            if wasp_dir.exists():
                for img_path in wasp_dir.glob('*.jpg'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(0)  # NOT Asian Hornet
                for img_path in wasp_dir.glob('*.jpeg'):
                    self.image_paths.append(str(img_path))
                    self.labels.append(0)  # NOT Asian Hornet

        print(f"Loaded {len(self.image_paths)} images")

        # Print class distribution
        labels_array = np.array(self.labels)
        n_asian = np.sum(labels_array == 1)
        n_not_asian = np.sum(labels_array == 0)
        print(f"  Asian Hornets: {n_asian} ({100*n_asian/len(labels_array):.1f}%)")
        print(f"  NOT Asian Hornets: {n_not_asian} ({100*n_not_asian/len(labels_array):.1f}%)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback for corrupted images
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def load_model():
    """Load trained binary model"""
    print(f"Loading model from {MODEL_PATH}...")
    model = models.efficientnet_b3(pretrained=False)

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 1)  # Single output for binary classification
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
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

            # Get single logit output
            logits = model(inputs).squeeze()  # Shape: [batch_size]

            # Apply temperature scaling and sigmoid
            probs_positive = torch.sigmoid(logits / TEMPERATURE)  # Probability of Asian Hornet

            # Convert to 2-class probability format for plotting functions
            # [prob_NOT_asian, prob_asian]
            probs_negative = 1 - probs_positive
            probs = torch.stack([probs_negative, probs_positive], dim=1)

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


def create_folder(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_ROC_AUC_multiple_class(classes, y_true, y_pred_prob, plot_dir=None, filename=None, show=False,
                                plot_and_save=True, replot=False):
    """Plot ROC-AUC curves for binary classification"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    threshold = dict()

    for i in range(len(classes)):
        fpr[i], tpr[i], threshold[i] = sklearn.metrics.roc_curve(to_categorical(y_true)[:, i], y_pred_prob[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], threshold["micro"] = sklearn.metrics.roc_curve(to_categorical(y_true).ravel(),
                                                                               y_pred_prob.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

    if plot_and_save:
        create_folder(plot_dir)
        with open(os.path.join(plot_dir, filename + '_roc_micro_' + '%.4f' % roc_auc["micro"] + '.txt'),
                  "w") as text_file:
            print(roc_auc, file=text_file)

    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    if plot_and_save:
        png_file = os.path.join(plot_dir, filename + '_ROC_AUC_' + '%.4f' % roc_auc["micro"] + '.pdf')

        if not os.path.exists(png_file) or replot:
            plt.clf()
            plt.close()
            plt.figure(figsize=(8, 8))
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.3f})'\
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.3f})'\
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            for i in range(len(classes)):
                plt.plot(fpr[i], tpr[i], lw=lw,
                         label='{0} (area = {1:0.3f})'\
                               ''.format(classes[i], roc_auc[i]))

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
    """Plot Precision-Recall curves for binary classification"""
    precision = dict()
    recall = dict()
    threshold = dict()
    average_precision = dict()
    Y_test = to_categorical(y_true)
    y_score = y_pred_prob

    n_classes = len(classes)

    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(Y_test[:, i],
                                                                       y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], threshold["micro"] = precision_recall_curve(Y_test.ravel(),
                                                                                     y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")

    create_folder(plot_dir)
    with open(os.path.join(plot_dir, filename + '_pr_' + '%.4f' % average_precision["micro"] + '.txt'),
              "w") as text_file:
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
        labels.append('micro-average (area = {0:0.3f})'\
                      ''.format(average_precision["micro"]))

        for i in range(n_classes):
            l, = plt.plot(recall[i], precision[i], lw=2)
            lines.append(l)
            labels.append('{0} (area = {1:0.3f})'\
                          ''.format(classes[i], average_precision[i]))

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


def plot_confusion_matrix(cm, classes, filename=None,
                          normalize=False, cmap=plt.cm.Blues, figsize=(8, 8), replot=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

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

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
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
    print("=" * 70)
    print("BINARY MODEL EVALUATION WITH PROFESSIONAL PLOTS")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Temperature scaling: T={TEMPERATURE}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Data sources (same as multiclass)
    data_sources = {
        'bees_hornets1': r'D:\Ultimate Dataset\BeesAndHornets1\Bee And Asian Hornet Detection',
        'bees_hornets2': r'D:\Ultimate Dataset\BeesAndHornets2\Dataset',
        'gbif_european_hornets': r'D:\Ultimate Dataset\european_hornets_gbif',
        'gbif_wasps': r'D:\Ultimate Dataset\wasps_gbif'
    }

    # Load dataset
    dataset = BinaryDataset(data_sources, split='valid', transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load model
    model = load_model()

    # Evaluate
    y_true, y_pred_prob = evaluate_model(model, dataloader)

    # Calculate accuracy
    y_pred_hard = np.argmax(y_pred_prob, axis=1)
    accuracy = np.mean(y_true == y_pred_hard) * 100

    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PROFESSIONAL PLOTS")
    print("=" * 70)

    plot_dir = str(OUTPUT_DIR)
    model_name = "binary"

    # 1. ROC-AUC curves
    print("\n1. Generating ROC-AUC curves...")
    roc_auc = plot_ROC_AUC_multiple_class(
        classes=CLASS_NAMES,
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        plot_dir=plot_dir,
        filename=model_name,
        show=False,
        plot_and_save=True,
        replot=True
    )
    print("   ROC-AUC scores:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"     {class_name}: {roc_auc[i]:.4f}")
    print(f"     Micro-average: {roc_auc['micro']:.4f}")
    print(f"     Macro-average: {roc_auc['macro']:.4f}")

    # 2. Precision-Recall curves
    print("\n2. Generating Precision-Recall curves...")
    avg_precision = plot_PR_curve_AUC_multiple_class(
        classes=CLASS_NAMES,
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        plot_dir=plot_dir,
        filename=model_name,
        show=False,
        replot=True
    )
    print("   Average Precision scores:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"     {class_name}: {avg_precision[i]:.4f}")
    print(f"     Micro-average: {avg_precision['micro']:.4f}")

    # 3. Confusion matrices
    print("\n3. Generating confusion matrices...")
    cm = confusion_matrix(y_true, y_pred_hard)

    figsize = (8, 8)
    output_confusion_file = os.path.join(plot_dir, model_name + '_cm.png')
    plot_confusion_matrix(cm, CLASS_NAMES, output_confusion_file,
                          normalize=False, figsize=figsize, replot=True)
    print(f"   Raw confusion matrix saved: {output_confusion_file}")

    output_confusion_file = os.path.join(plot_dir, model_name + '_cm_normalize.png')
    plot_confusion_matrix(cm, CLASS_NAMES, output_confusion_file,
                          normalize=True, figsize=figsize, replot=True)
    print(f"   Normalized confusion matrix saved: {output_confusion_file}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - ROC-AUC plot (PDF)")
    print("  - Precision-Recall plot (PDF)")
    print("  - Confusion matrix - raw counts (PNG)")
    print("  - Confusion matrix - normalized % (PNG)")
    print("  - Metric text files")
    print("=" * 70)


if __name__ == "__main__":
    main()
