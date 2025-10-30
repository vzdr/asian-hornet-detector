"""
Train one-class classifier for hornet detection.

Since all samples contain hornets, we use one-class classification to learn
what "typical hornet audio" looks like. This allows the model to flag
novel/unusual sounds as potential non-hornet audio.
"""

import numpy as np
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import json
import pickle

def load_features(features_dir):
    """Load embeddings and metadata."""
    features_dir = Path(features_dir)

    # Load embeddings
    embeddings_file = features_dir / "embeddings.npz"
    data = np.load(embeddings_file)
    embeddings = data['embeddings']

    # Load metadata
    metadata_file = features_dir / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded {len(embeddings)} embeddings of shape {embeddings.shape}")

    return embeddings, metadata


def group_by_video(metadata):
    """
    Group clips by source video for leave-one-video-out cross-validation.

    Returns:
        video_groups: dict mapping video_id to list of clip indices
    """
    video_groups = {}

    for idx, clip_meta in enumerate(metadata):
        # Extract video identifier from pair_id (e.g., "pair_0" -> 0)
        pair_id = clip_meta['pair_id']

        if pair_id not in video_groups:
            video_groups[pair_id] = []
        video_groups[pair_id].append(idx)

    return video_groups


def train_one_class_svm(X_train, contamination=0.1):
    """
    Train One-Class SVM.

    Args:
        X_train: Training features (n_samples, n_features)
        contamination: Expected proportion of outliers (0.0 to 0.5)

    Returns:
        model: Trained One-Class SVM
    """
    # One-Class SVM with RBF kernel
    model = OneClassSVM(
        kernel='rbf',
        gamma='scale',  # Auto-scale based on features
        nu=contamination  # Upper bound on fraction of outliers
    )

    model.fit(X_train)

    return model


def train_isolation_forest(X_train, contamination=0.1):
    """
    Train Isolation Forest.

    Args:
        X_train: Training features
        contamination: Expected proportion of outliers

    Returns:
        model: Trained Isolation Forest
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )

    model.fit(X_train)

    return model


def evaluate_one_class(model, X_test):
    """
    Evaluate one-class model.

    Returns:
        predictions: 1 for inliers (normal hornets), -1 for outliers
        scores: Anomaly scores (higher = more normal)
    """
    predictions = model.predict(X_test)

    # Get decision scores (distance from boundary)
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test)
    else:
        scores = model.score_samples(X_test)

    return predictions, scores


def cross_validate_one_class(embeddings, metadata, model_type='svm', contamination=0.1):
    """
    Perform leave-one-video-out cross-validation.

    Args:
        embeddings: Feature matrix (n_samples, n_features)
        metadata: Clip metadata
        model_type: 'svm' or 'isolation_forest'
        contamination: Expected outlier proportion

    Returns:
        results: Dictionary with cross-validation results
    """
    # Group clips by video
    video_groups = group_by_video(metadata)
    video_ids = list(video_groups.keys())

    print(f"\nFound {len(video_ids)} videos for cross-validation")
    print(f"Videos: {video_ids}")

    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    fold_results = []

    # Leave-one-video-out cross-validation
    for test_video_id in video_ids:
        print(f"\n{'='*60}")
        print(f"Fold: Testing on {test_video_id}")
        print(f"{'='*60}")

        # Split train/test by video
        test_indices = video_groups[test_video_id]
        train_indices = []
        for vid in video_ids:
            if vid != test_video_id:
                train_indices.extend(video_groups[vid])

        X_train = embeddings_scaled[train_indices]
        X_test = embeddings_scaled[test_indices]

        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        # Train model
        if model_type == 'svm':
            model = train_one_class_svm(X_train, contamination=contamination)
        else:
            model = train_isolation_forest(X_train, contamination=contamination)

        # Evaluate
        predictions, scores = evaluate_one_class(model, X_test)

        # Calculate metrics
        n_inliers = np.sum(predictions == 1)
        n_outliers = np.sum(predictions == -1)
        inlier_ratio = n_inliers / len(predictions)

        print(f"\nResults:")
        print(f"  Inliers (normal): {n_inliers} ({inlier_ratio*100:.1f}%)")
        print(f"  Outliers (anomalous): {n_outliers} ({(1-inlier_ratio)*100:.1f}%)")
        print(f"  Mean anomaly score: {np.mean(scores):.3f}")
        print(f"  Std anomaly score: {np.std(scores):.3f}")

        fold_results.append({
            'test_video': test_video_id,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_inliers': int(n_inliers),
            'n_outliers': int(n_outliers),
            'inlier_ratio': float(inlier_ratio),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'predictions': predictions.tolist(),
            'scores': scores.tolist()
        })

    # Overall statistics
    avg_inlier_ratio = np.mean([r['inlier_ratio'] for r in fold_results])

    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Average inlier ratio across folds: {avg_inlier_ratio*100:.1f}%")
    print(f"Model type: {model_type}")
    print(f"Contamination parameter: {contamination}")

    results = {
        'model_type': model_type,
        'contamination': contamination,
        'n_folds': len(video_ids),
        'fold_results': fold_results,
        'avg_inlier_ratio': float(avg_inlier_ratio)
    }

    return results, scaler


def train_final_model(embeddings, model_type='svm', contamination=0.1):
    """
    Train final model on all data for deployment.

    Returns:
        model: Trained model
        scaler: Fitted scaler
    """
    print(f"\n{'='*60}")
    print(f"Training final model on all data")
    print(f"{'='*60}")

    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Train
    if model_type == 'svm':
        model = train_one_class_svm(embeddings_scaled, contamination=contamination)
    else:
        model = train_isolation_forest(embeddings_scaled, contamination=contamination)

    print(f"Model trained on {len(embeddings)} samples")

    return model, scaler


def main():
    """Main training pipeline."""

    print("="*70)
    print("One-Class Classification for Hornet Detection")
    print("="*70)

    # Load features
    features_dir = Path(r"C:\Users\Zdravkovic\Desktop\hornet_detection\features\combined")
    embeddings, metadata = load_features(features_dir)

    # Try both model types
    for model_type in ['isolation_forest', 'svm']:
        print(f"\n\n{'#'*70}")
        print(f"# Training with {model_type.upper()}")
        print(f"{'#'*70}")

        # Cross-validation
        results, scaler = cross_validate_one_class(
            embeddings,
            metadata,
            model_type=model_type,
            contamination=0.1  # Expect 10% outliers
        )

        # Save results
        output_dir = Path(r"C:\Users\Zdravkovic\Desktop\hornet_detection\models")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / f"{model_type}_cv_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    # Train final model (using best performing one - Isolation Forest typically better)
    print(f"\n\n{'#'*70}")
    print(f"# Training final deployment model")
    print(f"{'#'*70}")

    final_model, final_scaler = train_final_model(
        embeddings,
        model_type='isolation_forest',
        contamination=0.1
    )

    # Save final model
    model_file = output_dir / "hornet_detector_final.pkl"
    scaler_file = output_dir / "scaler_final.pkl"

    with open(model_file, 'wb') as f:
        pickle.dump(final_model, f)

    with open(scaler_file, 'wb') as f:
        pickle.dump(final_scaler, f)

    print(f"\nFinal model saved to: {model_file}")
    print(f"Scaler saved to: {scaler_file}")

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
