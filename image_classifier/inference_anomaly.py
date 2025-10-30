"""
Anomaly Detection Inference - Asian Hornet Detector
Properly rejects out-of-distribution samples (monkeys, other insects, etc.)
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pickle
import numpy as np


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


def load_detector(model_path, scaler_path):
    """Load trained anomaly detector and scaler."""
    with open(model_path, 'rb') as f:
        detector = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return detector, scaler


def predict_anomaly(image_path, detector, scaler, feature_extractor):
    """
    Predict if image is an Asian hornet or not.

    Returns:
    - is_asian_hornet: bool
    - confidence: float (0-100%)
    - anomaly_score: float (raw score from detector)
    """
    # Extract features
    features = feature_extractor.extract_features(image_path)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict
    prediction = detector.predict(features_scaled)[0]
    # Get anomaly score (higher = more normal/inlier)
    anomaly_score = detector.decision_function(features_scaled)[0]

    # Convert prediction to boolean
    is_asian_hornet = (prediction == 1)

    # Convert anomaly score to confidence percentage
    # This is a heuristic - you can tune this based on your validation set
    # Positive scores = inlier, negative = outlier
    # We'll use a sigmoid-like transformation
    confidence = 1 / (1 + np.exp(-anomaly_score))  # Maps to [0, 1]
    confidence = confidence * 100  # Convert to percentage

    return is_asian_hornet, confidence, anomaly_score


def main():
    print("=" * 70)
    print("ASIAN HORNET ANOMALY DETECTOR - INFERENCE")
    print("=" * 70)

    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: py -3.10 inference_anomaly.py <path_to_image> [method]")
        print("\nArguments:")
        print("  <path_to_image>  - Path to the image to analyze")
        print("  [method]         - 'isolation_forest' or 'one_class_svm' (default: isolation_forest)")
        print("\nExamples:")
        print("  py -3.10 inference_anomaly.py C:\\Users\\test\\hornet.jpg")
        print("  py -3.10 inference_anomaly.py C:\\Users\\test\\hornet.jpg one_class_svm")
        print()
        sys.exit(1)

    image_path = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'isolation_forest'

    if method not in ['isolation_forest', 'one_class_svm']:
        print(f"\nError: Unknown method '{method}'")
        print("Valid methods: 'isolation_forest', 'one_class_svm'")
        sys.exit(1)

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image not found: {image_path}")
        sys.exit(1)

    # Model paths
    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / f'anomaly_detector_{method}.pkl'
    scaler_path = model_dir / f'scaler_{method}.pkl'

    if not model_path.exists() or not scaler_path.exists():
        print(f"\nError: Model not found for method '{method}'")
        print(f"Expected files:")
        print(f"  {model_path}")
        print(f"  {scaler_path}")
        print("\nPlease train the model first by running:")
        print("  py -3.10 train_anomaly_detector.py")
        sys.exit(1)

    # Load model
    print(f"\nLoading {method} model...")
    detector, scaler = load_detector(model_path, scaler_path)
    print("Model loaded successfully!")

    # Create feature extractor
    print("Loading feature extractor (ResNet50)...")
    feature_extractor = FeatureExtractor(device='cpu')
    print("Feature extractor loaded!")

    # Predict
    print(f"\nAnalyzing image: {image_path}")
    is_asian_hornet, confidence, anomaly_score = predict_anomaly(
        image_path, detector, scaler, feature_extractor
    )

    # Display results
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)

    if is_asian_hornet:
        print("\n  RESULT: ASIAN HORNET DETECTED")
        print(f"  Confidence: {confidence:.2f}%")
        status_symbol = "[✓]" if confidence > 70 else "[?]"
        print(f"  Status: {status_symbol}")
    else:
        print("\n  RESULT: NOT AN ASIAN HORNET")
        print(f"  Confidence: {100 - confidence:.2f}% (that it's NOT an Asian hornet)")
        print(f"  Status: [✗] REJECTED")

    print(f"\n  Raw Anomaly Score: {anomaly_score:.4f}")
    print(f"  Method Used: {method}")

    # Interpretation guide
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("\nAnomaly Score:")
    print("  Positive values = Detected as Asian hornet")
    print("  Negative values = Rejected (not Asian hornet)")
    print("  Larger magnitude = Higher confidence")

    print("\nConfidence Levels:")
    print("  > 90% : Very confident")
    print("  70-90%: Confident")
    print("  50-70%: Uncertain (manual verification recommended)")
    print("  < 50% : Low confidence (likely not Asian hornet)")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if is_asian_hornet and confidence > 90:
        print("\n  Very confident this IS an Asian hornet.")
        print("  Proceed with monitoring/control measures.")
    elif is_asian_hornet and confidence > 70:
        print("\n  Likely an Asian hornet, but some uncertainty.")
        print("  Recommend manual verification by expert.")
    elif is_asian_hornet and confidence <= 70:
        print("\n  Detection uncertain.")
        print("  Recommend manual verification - could be European hornet or wasp.")
    else:
        print("\n  This is NOT an Asian hornet.")
        print("  Possible reasons:")
        print("    - Different insect species")
        print("    - Not an insect at all")
        print("    - Poor image quality")

    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    main()
