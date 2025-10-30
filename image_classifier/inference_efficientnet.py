"""
EfficientNet-B3 Binary Asian Hornet Detector - Inference
Improved model for better accuracy
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json


def load_efficientnet_model(model_path):
    """Load trained EfficientNet-B3 model from checkpoint."""
    # Create model architecture
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
    image_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()

    # Decision
    is_asian_hornet = probability > threshold

    # Confidence
    if is_asian_hornet:
        confidence = probability * 100
    else:
        confidence = (1 - probability) * 100

    return is_asian_hornet, confidence, probability


def main():
    print("=" * 70)
    print("EFFICIENTNET-B3 ASIAN HORNET DETECTOR")
    print("=" * 70)

    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: py -3.10 inference_efficientnet.py <path_to_image> [threshold]")
        print("\nArguments:")
        print("  <path_to_image>  - Path to the image to analyze")
        print("  [threshold]      - Confidence threshold (0.0-1.0, default: from training)")
        print("\nExamples:")
        print("  py -3.10 inference_efficientnet.py C:\\Users\\test\\hornet.jpg")
        print("  py -3.10 inference_efficientnet.py C:\\Users\\test\\hornet.jpg 0.65")
        print()
        sys.exit(1)

    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else None

    if threshold is not None and (threshold < 0 or threshold > 1):
        print(f"\nError: Threshold must be between 0.0 and 1.0, got {threshold}")
        sys.exit(1)

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image not found: {image_path}")
        sys.exit(1)

    # Model path
    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "best_binary_efficientnet.pth"
    results_path = model_dir / "binary_efficientnet_results.json"

    if not model_path.exists():
        print(f"\nError: EfficientNet model not found: {model_path}")
        print("Please train the model first by running:")
        print("  py -3.10 train_binary_improved.py")
        sys.exit(1)

    # Load optimal threshold if available and not specified
    if threshold is None and results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
            if 'optimal_threshold' in results:
                threshold = results['optimal_threshold']
                print(f"\nUsing optimal threshold from training: {threshold:.2f}")
            else:
                threshold = 0.5
                print(f"\nUsing default threshold: {threshold:.2f}")
    elif threshold is None:
        threshold = 0.5
        print(f"\nUsing default threshold: {threshold:.2f}")

    # Load model
    print(f"\nLoading EfficientNet-B3 model...")
    model = load_efficientnet_model(model_path)
    print("Model loaded successfully!")

    # Image transform
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
