"""
Hornet Image Classifier - Inference Script
Test trained model on new images
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json


def load_model(model_path, num_classes=3):
    """Load trained model from checkpoint."""
    # Create model architecture
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint['class_names']


def predict_image(image_path, model, class_names, transform):
    """Predict class for a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    return {
        'predicted_class': class_names[predicted_class],
        'confidence': float(probabilities[predicted_class]),
        'all_probabilities': {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }
    }


def main():
    print("=" * 70)
    print("HORNET IMAGE CLASSIFIER - INFERENCE")
    print("=" * 70)

    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage: py -3.10 inference.py <path_to_image>")
        print("\nExample:")
        print("  py -3.10 inference.py C:\\Users\\test\\hornet.jpg")
        print()
        sys.exit(1)

    image_path = sys.argv[1]

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\nError: Image not found: {image_path}")
        sys.exit(1)

    # Model path
    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "best_model.pth"

    if not model_path.exists():
        print(f"\nError: Model not found: {model_path}")
        print("Please train the model first by running train_classifier.py")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model, class_names = load_model(model_path)
    print(f"Model loaded successfully!")
    print(f"Classes: {', '.join(class_names)}")

    # Image transform (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Predict
    print(f"\nAnalyzing image: {image_path}")
    result = predict_image(image_path, model, class_names, transform)

    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"\nPredicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")

    print("\nAll Class Probabilities:")
    for class_name, prob in result['all_probabilities'].items():
        bar_length = int(prob * 40)
        bar = "#" * bar_length + "-" * (40 - bar_length)
        print(f"  {class_name:20} {bar} {prob*100:6.2f}%")

    print("\n" + "=" * 70)

    # Species information
    print("\nSpecies Information:")
    species_info = {
        'Vespa_velutina': 'Asian Hornet (Target species - invasive)',
        'Vespa_crabro': 'European Hornet (Native species)',
        'Vespula_sp': 'Common Wasp species'
    }

    predicted_species = result['predicted_class']
    if predicted_species in species_info:
        print(f"  {species_info[predicted_species]}")

    print()


if __name__ == "__main__":
    main()
