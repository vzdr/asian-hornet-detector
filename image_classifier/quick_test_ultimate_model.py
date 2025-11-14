"""
Quick test of the saved ultimate model on a small subset
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np

def load_model(model_path):
    """Load the saved ultimate model"""
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

def test_on_samples():
    """Test on a few samples from each class"""
    model_path = Path(__file__).parent / "models" / "best_ultimate_efficientnet.pth"
    test_root = Path(r"D:\Ultimate Dataset\test_organized\small")

    print("Loading model...")
    model = load_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\nTesting on sample images...")
    print("="*70)

    classes = {
        'asian_hornets': 1,
        'bees': 0,
        'european_hornets': 0,
        'wasps': 0
    }

    for class_name, true_label in classes.items():
        class_dir = test_root / class_name
        if not class_dir.exists():
            continue

        # Test on first 10 images
        images = list(class_dir.glob('*.jpg'))[:10] + list(class_dir.glob('*.png'))[:10]
        images = images[:10]

        correct = 0
        probs = []

        for img_path in images:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image_tensor)
                prob = torch.sigmoid(output).item()
                pred = 1 if prob > 0.5 else 0

            probs.append(prob)
            if pred == true_label:
                correct += 1

        avg_prob = np.mean(probs)
        accuracy = correct / len(images) * 100

        print(f"{class_name:20} - Accuracy: {accuracy:5.1f}%  Mean prob: {avg_prob:.3f}  ({correct}/{len(images)})")

    print("="*70)
    print("\nModel test complete!")
    print("\nIf bee accuracy is >80%, the model is working!")
    print("If bee accuracy is <20%, the model has the same problem as before.")

if __name__ == "__main__":
    test_on_samples()
