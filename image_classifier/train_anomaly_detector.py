"""
Anomaly Detection for Asian Hornet (Vespa velutina)
Uses ResNet50 feature extraction + One-Class SVM/Isolation Forest
Properly rejects out-of-distribution samples (other insects, objects, etc.)
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pickle
import time
from tqdm import tqdm


class FeatureExtractor:
    """Extract deep features from ResNet50 pre-trained model."""

    def __init__(self, device='cpu'):
        self.device = device
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Remove final classification layer - use features before that
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.eval()
        self.feature_extractor.to(device)

        # Image preprocessing (same as ImageNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path):
        """Extract 2048-dimensional feature vector from image."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
            # Flatten from (1, 2048, 1, 1) to (2048,)
            features = features.squeeze().cpu().numpy()

        return features


def load_images_and_extract_features(data_root, feature_extractor, split='train'):
    """
    Load images and extract features.

    Returns:
    - features: numpy array of shape (n_samples, 2048)
    - labels: list of class labels (0=Vespa_crabro, 1=Vespa_velutina, 2=Vespula_sp)
    - paths: list of image paths
    """
    data_root = Path(data_root)
    class_names = ['Vespa_crabro', 'Vespa_velutina', 'Vespula_sp']

    all_features = []
    all_labels = []
    all_paths = []

    print(f"\nExtracting features from {split} set...")

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_root / split / 'images' / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue

        image_paths = list(class_dir.glob('*.jpg'))
        print(f"  {class_name}: {len(image_paths)} images")

        for img_path in tqdm(image_paths, desc=f"  {class_name}", leave=False):
            try:
                features = feature_extractor.extract_features(img_path)
                all_features.append(features)
                all_labels.append(class_idx)
                all_paths.append(str(img_path))
            except Exception as e:
                print(f"    Error processing {img_path}: {e}")

    features_array = np.array(all_features)
    print(f"  Extracted features shape: {features_array.shape}")

    return features_array, all_labels, all_paths


def train_anomaly_detector(X_train, method='isolation_forest', contamination=0.1):
    """
    Train anomaly detector on Asian hornet features only.

    Args:
    - X_train: Features from Asian hornets (Vespa_velutina)
    - method: 'isolation_forest' or 'one_class_svm'
    - contamination: Expected proportion of outliers in training data

    Returns:
    - detector: Trained anomaly detector
    - scaler: Feature scaler
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    print(f"\nTraining {method} on {len(X_train)} Asian hornet samples...")

    if method == 'isolation_forest':
        detector = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
    elif method == 'one_class_svm':
        detector = OneClassSVM(
            kernel='rbf',
            nu=contamination,
            gamma='auto'
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    detector.fit(X_scaled)

    return detector, scaler


def evaluate_detector(detector, scaler, X_test, y_test, target_class=1):
    """
    Evaluate anomaly detector.

    Args:
    - detector: Trained anomaly detector
    - scaler: Feature scaler
    - X_test: Test features
    - y_test: Test labels (0=Vespa_crabro, 1=Vespa_velutina, 2=Vespula_sp)
    - target_class: Class index for Asian hornet (1 = Vespa_velutina)

    Returns:
    - results: Dictionary with metrics
    """
    X_scaled = scaler.transform(X_test)

    # Predict: 1 = inlier (Asian hornet), -1 = outlier (not Asian hornet)
    predictions = detector.predict(X_scaled)

    # Get anomaly scores (higher = more normal/inlier)
    # For Isolation Forest: decision_function returns anomaly score
    # Positive = inlier, Negative = outlier
    scores = detector.decision_function(X_scaled)

    # Convert to binary labels for evaluation
    # True labels: 1 if Asian hornet, 0 otherwise
    y_true = (np.array(y_test) == target_class).astype(int)
    y_pred = (predictions == 1).astype(int)

    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    results = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'roc_auc': roc_auc,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }

    return results, y_true, y_pred, scores


def main():
    print("=" * 70)
    print("ASIAN HORNET ANOMALY DETECTOR - TRAINING")
    print("=" * 70)

    # Configuration
    DATA_ROOT = Path(r"C:\Users\Zdravkovic\Downloads\archive\data3000\data")
    MODEL_DIR = Path(__file__).parent / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    device = torch.device('cpu')
    print(f"\nDevice: {device}")

    # Step 1: Extract features from all images
    print("\n" + "=" * 70)
    print("STEP 1: FEATURE EXTRACTION")
    print("=" * 70)

    feature_extractor = FeatureExtractor(device=device)

    # Extract training features
    X_train_all, y_train_all, paths_train = load_images_and_extract_features(
        DATA_ROOT, feature_extractor, split='train'
    )

    # Extract validation features
    X_val_all, y_val_all, paths_val = load_images_and_extract_features(
        DATA_ROOT, feature_extractor, split='val'
    )

    # Step 2: Train on Asian hornets ONLY (class 1 = Vespa_velutina)
    print("\n" + "=" * 70)
    print("STEP 2: TRAIN ANOMALY DETECTOR")
    print("=" * 70)

    # Filter to get only Asian hornet samples for training
    asian_hornet_mask = np.array(y_train_all) == 1
    X_train_asian = X_train_all[asian_hornet_mask]

    print(f"\nTraining set breakdown:")
    print(f"  Total samples: {len(y_train_all)}")
    print(f"  Vespa_crabro (European hornet): {sum(np.array(y_train_all) == 0)}")
    print(f"  Vespa_velutina (Asian hornet): {sum(asian_hornet_mask)}")
    print(f"  Vespula_sp (Wasps): {sum(np.array(y_train_all) == 2)}")
    print(f"\nTraining anomaly detector on {len(X_train_asian)} Asian hornet samples only")

    # Train both methods for comparison
    methods = ['isolation_forest', 'one_class_svm']
    results_all = {}

    for method in methods:
        print(f"\n{'=' * 70}")
        print(f"METHOD: {method.upper()}")
        print(f"{'=' * 70}")

        start_time = time.time()

        detector, scaler = train_anomaly_detector(
            X_train_asian,
            method=method,
            contamination=0.05  # Expect 5% outliers in training data
        )

        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        results, y_true, y_pred, scores = evaluate_detector(
            detector, scaler, X_val_all, y_val_all, target_class=1
        )

        results['method'] = method
        results['train_time'] = train_time
        results['train_samples'] = len(X_train_asian)
        results['val_samples'] = len(y_val_all)

        results_all[method] = results

        # Print results
        print(f"\n{'=' * 70}")
        print(f"VALIDATION RESULTS - {method.upper()}")
        print(f"{'=' * 70}")
        print(f"Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"Precision: {results['precision']*100:.2f}%")
        print(f"Recall:    {results['recall']*100:.2f}%")
        print(f"F1-Score:  {results['f1_score']*100:.2f}%")
        print(f"ROC AUC:   {results['roc_auc']:.4f}")

        print("\nConfusion Matrix:")
        print("                  Predicted")
        print("                  NOT Asian  Asian")
        print(f"Actual NOT Asian  {results['confusion_matrix'][0][0]:6d}     {results['confusion_matrix'][0][1]:6d}")
        print(f"Actual Asian      {results['confusion_matrix'][1][0]:6d}     {results['confusion_matrix'][1][1]:6d}")

        # Calculate specific metrics
        tn, fp, fn, tp = results['confusion_matrix'][0][0], results['confusion_matrix'][0][1], \
                         results['confusion_matrix'][1][0], results['confusion_matrix'][1][1]

        print(f"\nBreakdown:")
        print(f"  True Positives (Correctly identified Asian hornets): {tp}")
        print(f"  True Negatives (Correctly rejected non-Asian): {tn}")
        print(f"  False Positives (Incorrectly identified as Asian): {fp}")
        print(f"  False Negatives (Missed Asian hornets): {fn}")

        # Save model
        model_path = MODEL_DIR / f'anomaly_detector_{method}.pkl'
        scaler_path = MODEL_DIR / f'scaler_{method}.pkl'

        with open(model_path, 'wb') as f:
            pickle.dump(detector, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

    # Save comprehensive results
    results_path = MODEL_DIR / 'anomaly_detection_results.json'

    # Convert numpy types to Python types for JSON serialization
    results_json = {}
    for method, res in results_all.items():
        results_json[method] = {
            'accuracy': float(res['accuracy']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'f1_score': float(res['f1_score']),
            'roc_auc': float(res['roc_auc']),
            'confusion_matrix': res['confusion_matrix'],
            'train_time': float(res['train_time']),
            'train_samples': int(res['train_samples']),
            'val_samples': int(res['val_samples'])
        }

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {results_path}")

    # Recommend best method
    best_method = max(results_all.keys(), key=lambda m: results_all[m]['f1_score'])
    print(f"\nRecommended method: {best_method.upper()}")
    print(f"  F1-Score: {results_all[best_method]['f1_score']*100:.2f}%")
    print(f"  ROC AUC: {results_all[best_method]['roc_auc']:.4f}")
    print()


if __name__ == "__main__":
    main()
