"""
Multi-Class Model Evaluation Script
===================================

Evaluates the trained 4-class model with:
- Overall accuracy
- Per-class precision, recall, F1-score
- 4x4 Confusion matrix visualization
- Comparison with validation set results

Usage:
    python evaluate_multiclass_model.py
    python evaluate_multiclass_model.py --model path/to/model.pth
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
import argparse

# Import dataset from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_multiclass_efficientnet import MultiClassDataset


def load_model(model_path, device):
    """Load trained model"""
    model = models.efficientnet_b3(pretrained=False)

    # Modify classifier for 4 classes
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 4)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions and labels"""
    all_preds = []
    all_labels = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(cm, class_names, save_path='multiclass_confusion_matrix.png'):
    """
    Plot confusion matrix with both counts and percentages
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix - Counts', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual Class', fontsize=12)
    axes[0].set_xlabel('Predicted Class', fontsize=12)

    # Percentages (normalized by row)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_title('Confusion Matrix - Percentages', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Actual Class', fontsize=12)
    axes[1].set_xlabel('Predicted Class', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_confidence_distribution(all_probs, all_labels, all_preds, class_names, save_path='confidence_distribution.png'):
    """Plot confidence distribution for correct and incorrect predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for class_idx in range(4):
        # Get predictions for this class
        class_mask = all_labels == class_idx
        class_probs = all_probs[class_mask][:, class_idx] * 100
        class_preds = all_preds[class_mask]

        correct_mask = class_preds == class_idx
        incorrect_mask = class_preds != class_idx

        correct_confidences = class_probs[correct_mask]
        incorrect_confidences = class_probs[incorrect_mask]

        # Plot histogram
        ax = axes[class_idx]
        bins = np.linspace(0, 100, 21)

        ax.hist(correct_confidences, bins=bins, alpha=0.7, label='Correct', color='green', edgecolor='black')
        ax.hist(incorrect_confidences, bins=bins, alpha=0.7, label='Incorrect', color='red', edgecolor='black')

        ax.set_title(f'{class_names[class_idx]}\nAccuracy: {100*len(correct_confidences)/(len(correct_confidences)+len(incorrect_confidences)):.1f}%',
                     fontweight='bold')
        ax.set_xlabel('Model Confidence (%)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confidence distribution saved to: {save_path}")
    plt.close()


def analyze_misclassifications(all_preds, all_labels, all_probs, class_names):
    """Analyze common misclassification patterns"""
    print("\n" + "="*70)
    print("MISCLASSIFICATION ANALYSIS")
    print("="*70)

    misclassified = all_preds != all_labels
    num_misclassified = misclassified.sum()

    print(f"\nTotal misclassifications: {num_misclassified} ({100*num_misclassified/len(all_labels):.2f}%)")

    if num_misclassified == 0:
        print("Perfect classification!")
        return

    # Analyze by class pair
    print("\nMost Common Confusions:")
    confusion_pairs = {}

    for i in range(len(all_labels)):
        if all_preds[i] != all_labels[i]:
            pair = (all_labels[i], all_preds[i])
            if pair not in confusion_pairs:
                confusion_pairs[pair] = []
            confusion_pairs[pair].append(all_probs[i][all_preds[i]] * 100)

    # Sort by frequency
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: len(x[1]), reverse=True)

    for (true_class, pred_class), confidences in sorted_pairs[:5]:
        avg_confidence = np.mean(confidences)
        print(f"  {class_names[true_class]:18} → {class_names[pred_class]:18}: "
              f"{len(confidences):4} cases (avg confidence: {avg_confidence:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-class model')
    parser.add_argument('--model', type=str, default='multiclass_models/best_multiclass_model.pth',
                        help='Path to model weights')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')

    args = parser.parse_args()

    print("="*70)
    print("MULTI-CLASS MODEL EVALUATION")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using: python train_multiclass_efficientnet.py")
        return

    # Data sources
    data_sources = {
        'bees_hornets1': r'D:\Ultimate Dataset\BeesAndHornets1\Bee And Asian Hornet Detection',
        'bees_hornets2': r'D:\Ultimate Dataset\BeesAndHornets2\Dataset',
        'gbif_european_hornets': r'D:\Ultimate Dataset\european_hornets_gbif',
        'gbif_wasps': r'D:\Ultimate Dataset\wasps_gbif'
    }

    # Load validation dataset
    print("\n" + "="*70)
    print("LOADING VALIDATION DATASET")
    print("="*70)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = MultiClassDataset(
        data_sources=data_sources,
        split='valid',
        transform=val_transform,
        samples_per_class=None  # Use all validation data
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    print(f"Model path: {model_path}")

    model = load_model(model_path, device)
    print("Model loaded successfully!")

    # Evaluate
    print("\n" + "="*70)
    print("RUNNING EVALUATION")
    print("="*70)

    all_preds, all_labels, all_probs = evaluate_model(model, val_loader, device)

    # Calculate metrics
    accuracy = 100 * accuracy_score(all_labels, all_preds)
    class_names = ['Asian Hornet', 'Bee', 'European Hornet', 'Wasp']

    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Classification report
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE")
    print("="*70)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print("Rows=Actual, Cols=Predicted")
    print(f"{'':20} {' '.join([f'{c:>15}' for c in class_names])}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:20} {' '.join([f'{v:>15}' for v in row])}")

    # Save confusion matrix plot
    output_dir = Path('evaluation_results')
    output_dir.mkdir(exist_ok=True)

    plot_confusion_matrix(cm, class_names, save_path=output_dir / 'confusion_matrix_multiclass.png')

    # Plot confidence distribution
    plot_confidence_distribution(all_probs, all_labels, all_preds, class_names,
                                 save_path=output_dir / 'confidence_distribution.png')

    # Analyze misclassifications
    analyze_misclassifications(all_preds, all_labels, all_probs, class_names)

    # Save results
    results = {
        'overall_accuracy': float(accuracy),
        'classification_report': classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels)
    }

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Confusion matrix: {output_dir / 'confusion_matrix_multiclass.png'}")
    print(f"  - Confidence distribution: {output_dir / 'confidence_distribution.png'}")
    print(f"  - JSON results: {output_dir / 'evaluation_results.json'}")


if __name__ == '__main__':
    main()
