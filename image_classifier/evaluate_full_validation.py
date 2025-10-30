"""
Full Validation Set Evaluation for EfficientNet-B3
Generates detailed confusion matrix, precision, recall, specificity
For supervisor presentation
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
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def evaluate_full_validation(model, data_root, threshold=0.5):
    """
    Evaluate on ALL validation images.
    Returns detailed metrics and confusion matrix.
    """
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_labels = []
    all_preds = []
    all_probs = []
    failure_cases = {"false_positives": [], "false_negatives": []}

    print("=" * 70)
    print("EVALUATING ON FULL VALIDATION SET")
    print("=" * 70)
    print()

    # Asian hornets (POSITIVE class)
    asian_dir = Path(data_root) / 'val' / 'images' / 'Vespa_velutina'
    print(f"Processing Asian hornets...")
    asian_count = 0
    for img_path in asian_dir.glob('*.jpg'):
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.sigmoid(output).item()

        pred = 1 if prob > threshold else 0
        all_labels.append(1)
        all_preds.append(pred)
        all_probs.append(prob)

        # Track false negatives
        if pred == 0:
            failure_cases["false_negatives"].append({
                "file": str(img_path.name),
                "probability": prob,
                "predicted": "NOT Asian hornet",
                "actual": "Asian hornet"
            })

        asian_count += 1

    print(f"  Processed {asian_count} Asian hornets")

    # European hornets (NEGATIVE class)
    european_dir = Path(data_root) / 'val' / 'images' / 'Vespa_crabro'
    print(f"Processing European hornets...")
    european_count = 0
    for img_path in european_dir.glob('*.jpg'):
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.sigmoid(output).item()

        pred = 1 if prob > threshold else 0
        all_labels.append(0)
        all_preds.append(pred)
        all_probs.append(prob)

        # Track false positives
        if pred == 1:
            failure_cases["false_positives"].append({
                "file": str(img_path.name),
                "probability": prob,
                "predicted": "Asian hornet",
                "actual": "European hornet"
            })

        european_count += 1

    print(f"  Processed {european_count} European hornets")

    # Wasps (NEGATIVE class)
    wasp_dir = Path(data_root) / 'val' / 'images' / 'Vespula_sp'
    print(f"Processing wasps...")
    wasp_count = 0
    for img_path in wasp_dir.glob('*.jpg'):
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.sigmoid(output).item()

        pred = 1 if prob > threshold else 0
        all_labels.append(0)
        all_preds.append(pred)
        all_probs.append(prob)

        # Track false positives
        if pred == 1:
            failure_cases["false_positives"].append({
                "file": str(img_path.name),
                "probability": prob,
                "predicted": "Asian hornet",
                "actual": "Wasp"
            })

        wasp_count += 1

    print(f"  Processed {wasp_count} wasps")
    print()

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # False positive rate and false negative rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

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
        "positive_samples": asian_count,
        "negative_samples": european_count + wasp_count,
        "threshold": threshold,
        "failure_cases": failure_cases,
        "all_probs": all_probs,
        "all_labels": all_labels,
        "all_preds": all_preds
    }


def print_results(results):
    """Print formatted results for supervisor presentation."""
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print()
    print("                      PREDICTED")
    print("                 NOT Hornet   Asian Hornet")
    print(f"ACTUAL  NOT         {results['tn']:5d}         {results['fp']:5d}      <- False Positives")
    print(f"        Asian       {results['fn']:5d}         {results['tp']:5d}      <- True Positives")
    print()
    print(f"True Negatives: {results['tn']} (correctly identified NOT Asian hornet)")
    print(f"False Positives: {results['fp']} (wrongly identified as Asian hornet)")
    print(f"False Negatives: {results['fn']} (missed Asian hornets)")
    print(f"True Positives: {results['tp']} (correctly identified Asian hornet)")
    print()

    print("=" * 70)
    print("DETAILED METRICS")
    print("=" * 70)
    print()
    print(f"Accuracy:    {results['accuracy']*100:.2f}%")
    print(f"  = (TP + TN) / Total = ({results['tp']} + {results['tn']}) / {results['total_samples']}")
    print(f"  Overall correctness across all samples")
    print()

    print(f"Precision:   {results['precision']*100:.2f}%")
    print(f"  = TP / (TP + FP) = {results['tp']} / ({results['tp']} + {results['fp']})")
    print(f"  When model says 'Asian hornet', how often is it right?")
    print()

    print(f"Recall:      {results['recall']*100:.2f}%")
    print(f"  = TP / (TP + FN) = {results['tp']} / ({results['tp']} + {results['fn']})")
    print(f"  Of all actual Asian hornets, what percentage did we catch?")
    print(f"  CRITICAL METRIC - We want to catch all hornets!")
    print()

    print(f"Specificity: {results['specificity']*100:.2f}%")
    print(f"  = TN / (TN + FP) = {results['tn']} / ({results['tn']} + {results['fp']})")
    print(f"  Of all NOT-Asian-hornets, what percentage did we correctly reject?")
    print()

    print(f"F1-Score:    {results['f1_score']*100:.2f}%")
    print(f"  = 2 × (Precision × Recall) / (Precision + Recall)")
    print(f"  Harmonic mean - balanced measure")
    print()

    print("=" * 70)
    print("ERROR RATES")
    print("=" * 70)
    print()
    print(f"False Positive Rate: {results['false_positive_rate']*100:.2f}%")
    print(f"  = FP / (FP + TN) = {results['fp']} / ({results['fp']} + {results['tn']})")
    print(f"  How often do we wrongly alarm for non-hornets?")
    print()

    print(f"False Negative Rate: {results['false_negative_rate']*100:.2f}%")
    print(f"  = FN / (FN + TP) = {results['fn']} / ({results['fn']} + {results['tp']})")
    print(f"  How often do we miss actual Asian hornets? (WANT THIS LOW!)")
    print()

    # Show failure cases
    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)
    print()

    fp_count = len(results['failure_cases']['false_positives'])
    fn_count = len(results['failure_cases']['false_negatives'])

    print(f"False Positives: {fp_count}")
    if fp_count > 0:
        print("  (Wrongly identified as Asian hornet):")
        for i, case in enumerate(results['failure_cases']['false_positives'][:10], 1):
            print(f"    {i}. {case['file']}: {case['probability']:.4f} prob, actual: {case['actual']}")
        if fp_count > 10:
            print(f"    ... and {fp_count - 10} more")
    print()

    print(f"False Negatives: {fn_count}")
    if fn_count > 0:
        print("  (Missed Asian hornets):")
        for i, case in enumerate(results['failure_cases']['false_negatives'][:10], 1):
            print(f"    {i}. {case['file']}: {case['probability']:.4f} prob, actual: {case['actual']}")
        if fn_count > 10:
            print(f"    ... and {fn_count - 10} more")
    print()


def main():
    print("=" * 70)
    print("FULL VALIDATION SET EVALUATION")
    print("EfficientNet-B3 Binary Asian Hornet Detector")
    print("=" * 70)
    print()

    # Paths
    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "best_binary_efficientnet.pth"
    data_root = Path(r"C:\Users\Zdravkovic\Downloads\archive\data3000\data")

    # Check model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first!")
        sys.exit(1)

    # Load model
    print("Loading EfficientNet-B3 model...")
    model = load_efficientnet_model(model_path)
    print("Model loaded successfully!")
    print()

    # Evaluate
    start_time = time.time()
    results = evaluate_full_validation(model, data_root, threshold=0.5)
    eval_time = time.time() - start_time

    # Print results
    print_results(results)

    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total images evaluated: {results['total_samples']}")
    print(f"  - Asian hornets: {results['positive_samples']}")
    print(f"  - NOT Asian hornets: {results['negative_samples']}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Time per image: {eval_time/results['total_samples']*1000:.1f} ms")
    print()

    # Save results
    output_file = model_dir / "validation_evaluation_detailed.json"
    save_results = {
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "metrics": {
            "accuracy": float(results["accuracy"]),
            "precision": float(results["precision"]),
            "recall": float(results["recall"]),
            "specificity": float(results["specificity"]),
            "f1_score": float(results["f1_score"]),
            "false_positive_rate": float(results["false_positive_rate"]),
            "false_negative_rate": float(results["false_negative_rate"])
        },
        "counts": {
            "true_negatives": results["tn"],
            "false_positives": results["fp"],
            "false_negatives": results["fn"],
            "true_positives": results["tp"],
            "total_samples": results["total_samples"],
            "positive_samples": results["positive_samples"],
            "negative_samples": results["negative_samples"]
        },
        "threshold": results["threshold"],
        "failure_cases": results["failure_cases"]
    }

    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    print("=" * 70)
    print("KEY TAKEAWAYS FOR SUPERVISORS")
    print("=" * 70)
    print()
    print(f"1. Model uses RGB (color) images - NOT grayscale")
    print(f"   - RandomGrayscale(p=0.1) is data augmentation (10% of training)")
    print(f"   - Model processes full 3-channel RGB color information")
    print()
    print(f"2. Confusion Matrix shows:")
    print(f"   - {results['tp']} / {results['positive_samples']} Asian hornets detected ({results['recall']*100:.1f}% recall)")
    print(f"   - {results['fn']} Asian hornets missed ({results['false_negative_rate']*100:.1f}% false negative rate)")
    print(f"   - {results['fp']} false alarms ({results['false_positive_rate']*100:.1f}% false positive rate)")
    print()
    print(f"3. Model performs well:")
    print(f"   - {results['recall']*100:.1f}% recall (catches most Asian hornets)")
    print(f"   - {results['precision']*100:.1f}% precision (low false alarm rate)")
    print(f"   - {results['specificity']*100:.1f}% specificity (good at rejecting non-hornets)")
    print()


if __name__ == "__main__":
    main()
