"""
Multi-Class Asian Hornet Inference Script
=========================================

Performs inference on images using the trained 4-class model.

Key features:
- Softmax activation for probability distribution across 4 classes
- Shows all class probabilities (not just top prediction)
- Natural confidence calibration (no more 0%/100% extremes)

Usage:
    python inference_multiclass_efficientnet.py path/to/image.jpg
    python inference_multiclass_efficientnet.py path/to/image.jpg --model path/to/model.pth
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
from pathlib import Path


class MultiClassInference:
    def __init__(self, model_path='multiclass_models/best_multiclass_model.pth', device=None):
        """
        Initialize multi-class inference

        Args:
            model_path: Path to trained model weights
            device: torch device (cuda/cpu), auto-detected if None
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Asian Hornet', 'Bee', 'European Hornet', 'Wasp']

        # Load model
        self.model = self._load_model(model_path)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"Inference ready on {self.device}")
        print(f"Model loaded from: {model_path}")

    def _load_model(self, model_path):
        """Load trained EfficientNet-B3 model"""
        # Build model architecture
        model = models.efficientnet_b3(pretrained=False)

        # Modify classifier for 4 classes
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 4)
        )

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        return model

    def predict(self, image_path, show_all_probs=True):
        """
        Predict class of an image

        Args:
            image_path: Path to image file
            show_all_probs: If True, show probabilities for all classes

        Returns:
            dict with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]  # Shape: (4,)

        # Get prediction
        predicted_idx = probabilities.argmax().item()
        predicted_class = self.class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item() * 100

        # Build results
        results = {
            'predicted_class': predicted_class,
            'predicted_idx': predicted_idx,
            'confidence': confidence,
            'probabilities': {
                self.class_names[i]: probabilities[i].item() * 100
                for i in range(4)
            }
        }

        return results

    def print_prediction(self, results):
        """Pretty print prediction results"""
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"\nPredicted Class: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.2f}%")
        print("\nAll Class Probabilities:")
        for class_name, prob in results['probabilities'].items():
            bar_length = int(prob / 2)  # Scale to 50 chars max
            bar = '█' * bar_length
            print(f"  {class_name:18} {prob:6.2f}% {bar}")
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Multi-class Asian hornet inference')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default='multiclass_models/best_multiclass_model.pth',
                        help='Path to model weights')
    parser.add_argument('--no-probs', action='store_true',
                        help='Only show top prediction (hide all probabilities)')

    args = parser.parse_args()

    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print(f"Please train the model first using: python train_multiclass_efficientnet.py")
        return

    # Run inference
    inference = MultiClassInference(model_path=str(model_path))
    results = inference.predict(str(image_path), show_all_probs=not args.no_probs)
    inference.print_prediction(results)

    # Interpretation
    confidence = results['confidence']
    predicted_class = results['predicted_class']

    print("Interpretation:")
    if confidence > 90:
        print(f"  ✓ Very confident this is a {predicted_class}")
    elif confidence > 70:
        print(f"  ✓ Likely a {predicted_class}")
    elif confidence > 50:
        print(f"  ? Probably a {predicted_class}, but uncertain")
    else:
        print(f"  ? Very uncertain - image may be ambiguous")
        # Show runner-up
        sorted_probs = sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True)
        runner_up, runner_prob = sorted_probs[1]
        print(f"    Runner-up: {runner_up} ({runner_prob:.2f}%)")


if __name__ == '__main__':
    main()
