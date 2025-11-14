"""
Comprehensive Evaluation on Ultimate Dataset
Tests existing EfficientNet-B3 model on small, medium, and large test sets
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time
import matplotlib.pyplot as plt
import seaborn as sns


def load_efficientnet_model(model_path):
    """Load trained EfficientNet-B3 model."""
    model = models.efficientnet_b3(pretrained=False)

    # Replace classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 1)
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def evaluate_on_test_set(model, test_root, threshold=0.5):
    """
    Evaluate model on a test set.

    Args:
        model: loaded PyTorch model
        test_root: path to test set with class subdirectories
        threshold: classification threshold

    Returns:
        dict: comprehensive evaluation results
    """
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_root = Path(test_root)
    all_labels = []
    all_preds = []
    all_probs = []
    class_wise_results = {}
    failure_cases = {"false_positives": [], "false_negatives": []}

    print(f"\nEvaluating on: {test_root.name}")
    print("="*70)

    start_time = time.time()
    total_images = 0

    # Classes: asian_hornets (POSITIVE), bees, european_hornets, wasps (all NEGATIVE)
    class_mapping = {
        'asian_hornets': 1,  # POSITIVE class
        'bees': 0,           # NEGATIVE class
        'european_hornets': 0,  # NEGATIVE class
        'wasps': 0           # NEGATIVE class
    }

    for class_name, label in class_mapping.items():
        class_dir = test_root / class_name
        if not class_dir.exists():
            print(f"  Skipping {class_name} (directory not found)")
            continue

        class_results = {'correct': 0, 'total': 0, 'probs': []}

        print(f"  Processing {class_name}...")
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(image_tensor)
                    prob = torch.sigmoid(output).item()

                pred = 1 if prob > threshold else 0
                all_labels.append(label)
                all_preds.append(pred)
                all_probs.append(prob)

                class_results['probs'].append(prob)
                class_results['total'] += 1
                if pred == label:
                    class_results['correct'] += 1

                # Track failure cases
                if label == 1 and pred == 0:  # False negative
                    failure_cases["false_negatives"].append({
                        "file": str(img_path.name),
                        "probability": prob,
                        "actual_class": class_name
                    })
                elif label == 0 and pred == 1:  # False positive
                    failure_cases["false_positives"].append({
                        "file": str(img_path.name),
                        "probability": prob,
                        "actual_class": class_name
                    })

                total_images += 1

                if total_images % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = total_images / elapsed
                    print(f"    Processed {total_images} images ({rate:.1f} img/sec)")

            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                continue

        # Store class-wise results
        class_acc = class_results['correct'] / class_results['total'] if class_results['total'] > 0 else 0
        class_wise_results[class_name] = {
            'accuracy': class_acc,
            'total': class_results['total'],
            'correct': class_results['correct'],
            'mean_prob': np.mean(class_results['probs']) if class_results['probs'] else 0
        }

        print(f"    {class_name}: {class_results['correct']}/{class_results['total']} "
              f"({class_acc*100:.2f}% accuracy, mean prob: {class_wise_results[class_name]['mean_prob']:.3f})")

    eval_time = time.time() - start_time

    # Calculate overall metrics
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Per-class false positive rates
    bee_fp_rate = 0
    european_fp_rate = 0
    wasp_fp_rate = 0

    if 'bees' in class_wise_results:
        bee_total = class_wise_results['bees']['total']
        bee_correct = class_wise_results['bees']['correct']
        bee_fp_rate = (bee_total - bee_correct) / bee_total if bee_total > 0 else 0

    if 'european_hornets' in class_wise_results:
        euro_total = class_wise_results['european_hornets']['total']
        euro_correct = class_wise_results['european_hornets']['correct']
        european_fp_rate = (euro_total - euro_correct) / euro_total if euro_total > 0 else 0

    if 'wasps' in class_wise_results:
        wasp_total = class_wise_results['wasps']['total']
        wasp_correct = class_wise_results['wasps']['correct']
        wasp_fp_rate = (wasp_total - wasp_correct) / wasp_total if wasp_total > 0 else 0

    print("\n" + "="*70)
    print("OVERALL METRICS")
    print("="*70)
    print(f"Accuracy:    {accuracy*100:.2f}%")
    print(f"Precision:   {precision*100:.2f}%")
    print(f"Recall:      {recall*100:.2f}%")
    print(f"Specificity: {specificity*100:.2f}%")
    print(f"F1-Score:    {f1*100:.2f}%")
    print(f"FPR:         {fpr*100:.2f}%")
    print(f"FNR:         {fnr*100:.2f}%")
    print(f"\nTotal images: {total_images}")
    print(f"Evaluation time: {eval_time:.2f} seconds ({total_images/eval_time:.1f} img/sec)")

    print("\n" + "="*70)
    print("PER-CLASS FALSE POSITIVE RATES")
    print("="*70)
    print(f"Bees FP rate:             {bee_fp_rate*100:.2f}%")
    print(f"European Hornets FP rate: {european_fp_rate*100:.2f}%")
    print(f"Wasps FP rate:            {wasp_fp_rate*100:.2f}%")

    return {
        "confusion_matrix": cm,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "total_samples": len(all_labels),
        "threshold": threshold,
        "eval_time": eval_time,
        "images_per_sec": total_images / eval_time,
        "class_wise_results": class_wise_results,
        "bee_fp_rate": bee_fp_rate,
        "european_fp_rate": european_fp_rate,
        "wasp_fp_rate": wasp_fp_rate,
        "failure_cases": failure_cases,
        "all_probs": all_probs,
        "all_labels": all_labels,
        "all_preds": all_preds
    }


def plot_confusion_matrix(cm, test_name, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NOT Asian Hornet', 'Asian Hornet'],
                yticklabels=['NOT Asian Hornet', 'Asian Hornet'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {test_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{test_name}.png', dpi=300)
    plt.close()


def main():
    print("="*70)
    print("COMPREHENSIVE EVALUATION ON ULTIMATE DATASET")
    print("="*70)

    # Paths
    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "best_balanced_30k_efficientnet.pth"  # 30k model
    test_root = Path(r"D:\Ultimate Dataset\test_organized")
    output_dir = Path(__file__).parent / "balanced_30k_evaluation_results"
    output_dir.mkdir(exist_ok=True)

    # Check model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Load model
    print("\nLoading EfficientNet-B3 model...")
    model = load_efficientnet_model(model_path)
    print("Model loaded successfully!")

    # Test on different dataset sizes
    test_sets = ['small', 'medium', 'large']
    all_results = {}

    for test_name in test_sets:
        test_path = test_root / test_name

        if not test_path.exists():
            print(f"\nSkipping {test_name} test (directory not found: {test_path})")
            continue

        # Evaluate
        results = evaluate_on_test_set(model, test_path, threshold=0.5)

        # Save results
        save_results = {
            "test_set": test_name,
            "confusion_matrix": results["confusion_matrix"].tolist(),
            "metrics": {
                "accuracy": float(results["accuracy"]),
                "precision": float(results["precision"]),
                "recall": float(results["recall"]),
                "specificity": float(results["specificity"]),
                "f1_score": float(results["f1_score"]),
                "false_positive_rate": float(results["false_positive_rate"]),
                "false_negative_rate": float(results["false_negative_rate"]),
                "bee_fp_rate": float(results["bee_fp_rate"]),
                "european_fp_rate": float(results["european_fp_rate"]),
                "wasp_fp_rate": float(results["wasp_fp_rate"])
            },
            "counts": {
                "true_negatives": results["tn"],
                "false_positives": results["fp"],
                "false_negatives": results["fn"],
                "true_positives": results["tp"],
                "total_samples": results["total_samples"]
            },
            "performance": {
                "eval_time_seconds": float(results["eval_time"]),
                "images_per_second": float(results["images_per_sec"])
            },
            "class_wise_results": results["class_wise_results"],
            "threshold": results["threshold"]
        }

        output_file = output_dir / f"{test_name}_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(save_results, f, indent=2)

        print(f"\n Results saved to: {output_file}")

        # Plot confusion matrix
        plot_confusion_matrix(results["confusion_matrix"], test_name, output_dir)

        all_results[test_name] = results

    # Create summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON ACROSS TEST SETS")
    print("="*70)
    print(f"{'Metric':<25} {'Small':<15} {'Medium':<15} {'Large':<15}")
    print("-"*70)

    metrics_to_compare = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'bee_fp_rate']

    for metric in metrics_to_compare:
        values = []
        for test_name in test_sets:
            if test_name in all_results:
                values.append(f"{all_results[test_name][metric]*100:.2f}%")
            else:
                values.append("N/A")

        print(f"{metric.replace('_', ' ').title():<25} {values[0]:<15} {values[1]:<15} {values[2]:<15}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
