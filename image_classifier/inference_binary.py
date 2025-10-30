"""
Binary Asian Hornet Detector - Inference
Simple output: "YES this is an Asian hornet (X% confidence)" or "NO (X% confidence)"
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json


def load_binary_model(model_path):
    """Load trained binary model from checkpoint."""
    # Create model architecture
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def predict_asian_hornet(image_path, model, transform, threshold=0.5):
    """
    Predict if image contains an Asian hornet.

    Returns:
    - is_asian_hornet: bool
    - confidence: float (0-100%)
    - raw_probability: float (0-1)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()  # Probability of being Asian hornet

    # Decision based on threshold
    is_asian_hornet = probability > threshold

    # Confidence is the distance from 0.5 (decision boundary)
    # If prob = 0.9, confidence in "YES" = 90%
    # If prob = 0.1, confidence in "NO" = 90%
    if is_asian_hornet:
        confidence = probability * 100
    else:
        confidence = (1 - probability) * 100

    return is_asian_hornet, confidence, probability


def main():
    print("=" * 70)
    print("BINARY ASIAN HORNET DETECTOR")
    print("=" * 70)

    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: py -3.10 inference_binary.py <path_to_image> [threshold]")
        print("\nArguments:")
        print("  <path_to_image>  - Path to the image to analyze")
        print("  [threshold]      - Confidence threshold (0.0-1.0, default: 0.5)")
        print("\nExamples:")
        print("  py -3.10 inference_binary.py C:\\Users\\test\\hornet.jpg")
        print("  py -3.10 inference_binary.py C:\\Users\\test\\hornet.jpg 0.7")
        print()
        sys.exit(1)

    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    if threshold < 0 or threshold > 1:
        print(f"\nError: Threshold must be between 0.0 and 1.0, got {threshold}")
        sys.exit(1)

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image not found: {image_path}")
        sys.exit(1)

    # Model path
    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "best_binary_model.pth"
    results_path = model_dir / "binary_training_results.json"

    if not model_path.exists():
        print(f"\nError: Model not found: {model_path}")
        print("Please train the model first by running:")
        print("  py -3.10 train_binary_detector.py")
        sys.exit(1)

    # Load optimal threshold if available
    if results_path.exists() and len(sys.argv) < 3:  # User didn't specify threshold
        with open(results_path, 'r') as f:
            results = json.load(f)
            if 'optimal_threshold' in results:
                threshold = results['optimal_threshold']
                print(f"\nUsing optimal threshold from training: {threshold:.2f}")

    # Load model
    print(f"\nLoading binary model...")
    model = load_binary_model(model_path)
    print("Model loaded successfully!")

    # Image transform (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Predict
    print(f"\nAnalyzing image: {image_path}")
    is_asian_hornet, confidence, probability = predict_asian_hornet(
        image_path, model, transform, threshold
    )

    # Display results
    print("\n" + "=" * 70)
    print("DETECTION RESULT")
    print("=" * 70)

    if is_asian_hornet:
        print("\n  [YES] ASIAN HORNET DETECTED")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Raw Probability: {probability:.4f}")

        if confidence > 90:
            print("\n  Status: HIGH CONFIDENCE")
            print("  Recommendation: Likely an Asian hornet")
        elif confidence > 75:
            print("\n  Status: MEDIUM CONFIDENCE")
            print("  Recommendation: Probably an Asian hornet, verify if possible")
        else:
            print("\n  Status: LOW CONFIDENCE")
            print("  Recommendation: Uncertain, manual verification recommended")

    else:
        print("\n  [NO] NOT AN ASIAN HORNET")
        print(f"  Confidence: {confidence:.2f}%")
        print(f"  Raw Probability: {probability:.4f}")

        if confidence > 90:
            print("\n  Status: HIGH CONFIDENCE")
            print("  Recommendation: Definitely NOT an Asian hornet")
        elif confidence > 75:
            print("\n  Status: MEDIUM CONFIDENCE")
            print("  Recommendation: Probably NOT an Asian hornet")
        else:
            print("\n  Status: LOW CONFIDENCE")
            print("  Recommendation: Uncertain, could be Asian hornet")

    # Interpretation guide
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"\nDecision Threshold: {threshold:.2f}")
    print(f"  Probabilities > {threshold:.2f} -> Asian hornet detected")
    print(f"  Probabilities <= {threshold:.2f} -> NOT Asian hornet")

    print("\nConfidence Levels:")
    print("  >90%: Very confident in classification")
    print("  75-90%: Confident")
    print("  60-75%: Moderate confidence")
    print("  <60%: Low confidence (near decision boundary)")

    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    main()
