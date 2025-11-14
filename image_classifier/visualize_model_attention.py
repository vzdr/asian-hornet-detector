"""
Grad-CAM Visualization: See what the model focuses on
Shows heatmaps highlighting which parts of the image the model uses for classification
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

class GradCAM:
    """Generate Grad-CAM heatmaps to visualize model attention"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """Generate Class Activation Map"""
        # Forward pass
        model_output = self.model(input_image)

        if target_class is None:
            target_class = torch.sigmoid(model_output)

        # Backward pass
        self.model.zero_grad()
        model_output.backward(torch.ones_like(model_output))

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Pool gradients across spatial dimensions
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # Apply ReLU
        cam = torch.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()


def load_model(model_path):
    """Load the trained model"""
    model = models.efficientnet_b3(pretrained=False)

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, 1)
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def apply_heatmap(image, cam, alpha=0.5):
    """Overlay heatmap on original image"""
    # Resize CAM to match image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlaid = (1 - alpha) * image + alpha * heatmap
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

    return overlaid, heatmap


def visualize_samples(model_path, test_root, output_dir, samples_per_class=5):
    """Generate Grad-CAM visualizations for sample images"""

    print("="*70)
    print("GRAD-CAM VISUALIZATION - MODEL ATTENTION ANALYSIS")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model = load_model(model_path)

    # Get target layer (last conv layer in EfficientNet)
    target_layer = model.features[-1]

    # Create Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Classes to visualize
    classes = {
        'asian_hornets': 'Asian Hornet (POSITIVE)',
        'bees': 'Bee (NEGATIVE)',
        'european_hornets': 'European Hornet (NEGATIVE)',
        'wasps': 'Wasp (NEGATIVE)'
    }

    for class_name, class_label in classes.items():
        class_dir = test_root / class_name
        if not class_dir.exists():
            continue

        print(f"\nProcessing {class_label}...")

        # Get sample images
        image_files = list(class_dir.glob('*.jpg'))[:samples_per_class]

        for idx, img_path in enumerate(image_files):
            try:
                # Load image
                original_image = Image.open(img_path).convert('RGB')
                original_np = np.array(original_image)

                # Transform for model
                input_tensor = transform(original_image).unsqueeze(0)

                # Generate prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()
                    prediction = "Asian Hornet" if prob > 0.5 else "NOT Asian Hornet"

                # Generate Grad-CAM
                input_tensor.requires_grad = True
                cam = grad_cam.generate_cam(input_tensor)

                # Resize original image to match model input
                original_resized = cv2.resize(original_np, (224, 224))

                # Apply heatmap
                overlaid, heatmap = apply_heatmap(original_resized, cam, alpha=0.4)

                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(original_resized)
                axes[0].set_title(f'Original\n{class_label}', fontsize=10, fontweight='bold')
                axes[0].axis('off')

                # Heatmap
                axes[1].imshow(heatmap)
                axes[1].set_title('Attention Heatmap\n(Red = High Attention)', fontsize=10, fontweight='bold')
                axes[1].axis('off')

                # Overlay
                axes[2].imshow(overlaid)
                axes[2].set_title(f'Overlay\nPrediction: {prediction}\nConfidence: {prob:.2%}',
                                 fontsize=10, fontweight='bold',
                                 color='green' if (prob > 0.5 and class_name == 'asian_hornets') or
                                                 (prob <= 0.5 and class_name != 'asian_hornets')
                                       else 'red')
                axes[2].axis('off')

                plt.suptitle(f'{class_label} - Sample {idx+1}', fontsize=14, fontweight='bold')
                plt.tight_layout()

                # Save
                output_file = output_dir / f'{class_name}_sample_{idx+1}.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"  Saved: {output_file.name} (Pred: {prediction}, Prob: {prob:.2%})")

            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nAll heatmaps saved to: {output_dir}")
    print("\nWhat to look for:")
    print("  - Red areas = High model attention (where it's focusing)")
    print("  - Blue areas = Low attention (ignored regions)")
    print("\nFor Asian Hornets, check if red focuses on:")
    print("  ✓ Orange band on torso")
    print("  ✓ Yellow legs")
    print("  ✓ Yellow face")
    print("\nFor European Hornets/Wasps misclassified as Asian:")
    print("  - See what features confused the model")
    print("  - May reveal need for more training data or feature engineering")


def main():
    model_path = Path(__file__).parent / "models" / "best_balanced_30k_efficientnet.pth"
    test_root = Path(r"D:\Ultimate Dataset\test_organized\small")  # Use small test for quick visualization
    output_dir = Path(__file__).parent / "gradcam_30k_visualizations"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    if not test_root.exists():
        print(f"Error: Test data not found at {test_root}")
        return

    visualize_samples(model_path, test_root, output_dir, samples_per_class=20)


if __name__ == "__main__":
    main()
