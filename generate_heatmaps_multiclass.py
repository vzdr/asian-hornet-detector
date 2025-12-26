"""
Multi-Class Grad-CAM Heatmap Generation
========================================

Generates Grad-CAM visualizations for the 4-class model showing:
- What the model focuses on for each class
- Probability distributions across all 4 classes
- 5 samples per class (20 total)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import random

# Configuration
MODEL_PATH = "multiclass_models/best_multiclass_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = ['Asian Hornet', 'Bee', 'European Hornet', 'Wasp']

# Output directory
OUTPUT_DIR = Path("heatmaps_multiclass")
OUTPUT_DIR.mkdir(exist_ok=True)


class GradCAMMultiClass:
    """Grad-CAM for multi-class classification"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class_idx):
        """
        Generate CAM heatmap for a specific class

        Args:
            input_image: Input tensor [1, 3, H, W]
            target_class_idx: Index of target class (0-3)

        Returns:
            cam: Heatmap numpy array [H, W]
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        # Zero gradients
        self.model.zero_grad()

        # Create one-hot target
        target = torch.zeros_like(output)
        target[0, target_class_idx] = 1

        # Backward pass
        output.backward(gradient=target, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted combination
        cam = (weights * activations).sum(dim=0)  # [H, W]

        # ReLU
        cam = F.relu(cam)

        # Normalize
        if cam.max() > 0:
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


def load_model():
    """Load trained 4-class model"""
    print(f"Loading model from {MODEL_PATH}...")
    model = models.efficientnet_b3(pretrained=False)

    # Modify classifier for 4 classes
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    return model


def get_sample_images():
    """Get 5 sample images from each class"""
    samples = {
        'asian_hornet': [],
        'bee': [],
        'european_hornet': [],
        'wasp': []
    }

    # Get Asian hornets and bees from YOLO datasets
    yolo_sources = [
        (Path(r'D:\Ultimate Dataset\BeesAndHornets1\Bee And Asian Hornet Detection\valid'),
         Path(r'D:\Ultimate Dataset\BeesAndHornets1\Bee And Asian Hornet Detection\valid\images')),
        (Path(r'D:\Ultimate Dataset\BeesAndHornets2\Dataset\valid'),
         Path(r'D:\Ultimate Dataset\BeesAndHornets2\Dataset\valid\images'))
    ]

    for labels_root, images_root in yolo_sources:
        labels_dir = labels_root / 'labels'
        if not labels_dir.exists() or not images_root.exists():
            continue

        for label_file in labels_dir.glob('*.txt'):
            if len(samples['asian_hornet']) >= 5 and len(samples['bee']) >= 5:
                break

            try:
                with open(label_file) as f:
                    line = f.readline().strip()
                    if not line:
                        continue

                    class_id = int(line.split()[0])

                    # Find corresponding image
                    image_path = images_root / f"{label_file.stem}.jpg"
                    if not image_path.exists():
                        image_path = images_root / f"{label_file.stem}.jpeg"

                    if not image_path.exists():
                        continue

                    # Class 0 = bee, Class 1 = asian hornet (in YOLO format)
                    if class_id == 0 and len(samples['bee']) < 5:
                        samples['bee'].append(image_path)
                    elif class_id == 1 and len(samples['asian_hornet']) < 5:
                        samples['asian_hornet'].append(image_path)
            except:
                continue

    # Get European hornets from GBIF
    european_dir = Path(r'D:\Ultimate Dataset\european_hornets_gbif')
    if european_dir.exists():
        european_images = list(european_dir.glob('*.jpg')) + list(european_dir.glob('*.jpeg'))
        samples['european_hornet'] = random.sample(european_images, min(5, len(european_images)))

    # Get wasps from GBIF
    wasp_dir = Path(r'D:\Ultimate Dataset\wasps_gbif')
    if wasp_dir.exists():
        wasp_images = list(wasp_dir.glob('*.jpg')) + list(wasp_dir.glob('*.jpeg'))
        samples['wasp'] = random.sample(wasp_images, min(5, len(wasp_images)))

    return samples


def create_heatmap_overlay(original_img, cam, alpha=0.4):
    """Create heatmap overlay on original image"""
    # Resize CAM to match image
    h, w = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)

    return overlay


def process_image(image_path, model, grad_cam, transform):
    """Process single image and generate all visualizations"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = probabilities.argmax().item()

    # Generate CAM for predicted class
    cam = grad_cam.generate_cam(image_tensor, predicted_class)

    # Create overlay
    overlay = create_heatmap_overlay(image_np, cam)

    # Get probability values
    probs = {CLASS_NAMES[i]: probabilities[i].item() * 100 for i in range(NUM_CLASSES)}

    return image_np, overlay, cam, predicted_class, probs


def create_visualization(image_np, overlay, cam, predicted_class, probs, true_class_name, save_path):
    """Create final visualization with all info"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title(f'Original Image\nTrue Class: {true_class_name}', fontsize=12)
    axes[0].axis('off')

    # Grad-CAM heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(overlay)
    title = f'Prediction: {CLASS_NAMES[predicted_class]}\nConfidence: {probs[CLASS_NAMES[predicted_class]]:.1f}%'
    axes[2].set_title(title, fontsize=12)
    axes[2].axis('off')

    # Add probability distribution as text
    prob_text = "Probabilities:\n"
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_probs:
        marker = " <--" if class_name == CLASS_NAMES[predicted_class] else ""
        prob_text += f"{class_name:18} {prob:5.1f}%{marker}\n"

    fig.text(0.5, -0.05, prob_text, ha='center', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_grid(all_results, save_path):
    """Create grid showing all samples"""
    n_classes = 4
    n_samples = 5

    fig, axes = plt.subplots(n_classes, n_samples * 3, figsize=(n_samples * 6, n_classes * 3))

    for class_idx, class_name in enumerate(CLASS_NAMES):
        if class_name.lower().replace(' ', '_') not in all_results:
            continue

        samples = all_results[class_name.lower().replace(' ', '_')]

        for sample_idx, (img, overlay, cam) in enumerate(samples[:n_samples]):
            col_base = sample_idx * 3

            # Original
            axes[class_idx, col_base].imshow(img)
            axes[class_idx, col_base].axis('off')
            if sample_idx == 0:
                axes[class_idx, col_base].set_ylabel(class_name, fontsize=14, fontweight='bold')
            if class_idx == 0:
                axes[class_idx, col_base].set_title('Original', fontsize=12)

            # CAM
            axes[class_idx, col_base + 1].imshow(cam, cmap='jet')
            axes[class_idx, col_base + 1].axis('off')
            if class_idx == 0:
                axes[class_idx, col_base + 1].set_title('Heatmap', fontsize=12)

            # Overlay
            axes[class_idx, col_base + 2].imshow(overlay)
            axes[class_idx, col_base + 2].axis('off')
            if class_idx == 0:
                axes[class_idx, col_base + 2].set_title('Overlay', fontsize=12)

    plt.suptitle('Grad-CAM Visualizations - Multi-Class Asian Hornet Classifier',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("="*70)
    print("MULTI-CLASS GRAD-CAM HEATMAP GENERATION")
    print("="*70)

    # Load model
    model = load_model()

    # Setup Grad-CAM (target last convolutional layer)
    target_layer = model.features[-1]
    grad_cam = GradCAMMultiClass(model, target_layer)

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get sample images
    print("\nFinding sample images...")
    samples = get_sample_images()

    for class_key, images in samples.items():
        print(f"  {class_key}: {len(images)} samples")

    # Process all images
    all_results = {}

    for class_key, class_name in [('asian_hornet', 'Asian Hornet'),
                                   ('bee', 'Bee'),
                                   ('european_hornet', 'European Hornet'),
                                   ('wasp', 'Wasp')]:

        if class_key not in samples or not samples[class_key]:
            print(f"\nWARNING: No samples found for {class_name}")
            continue

        print(f"\n{'='*70}")
        print(f"Processing {class_name} samples...")
        print(f"{'='*70}")

        all_results[class_key] = []

        for idx, image_path in enumerate(samples[class_key], 1):
            print(f"  [{idx}/{len(samples[class_key])}] {image_path.name}")

            try:
                image_np, overlay, cam, predicted_class, probs = process_image(
                    image_path, model, grad_cam, transform
                )

                # Save individual visualization
                save_path = OUTPUT_DIR / f"{class_key}_{idx}.png"
                create_visualization(image_np, overlay, cam, predicted_class, probs,
                                      class_name, save_path)

                # Store for grid
                all_results[class_key].append((image_np, overlay, cam))

                # Print prediction
                correct = (CLASS_NAMES[predicted_class] == class_name)
                status = "CORRECT" if correct else "INCORRECT"
                print(f"      Predicted: {CLASS_NAMES[predicted_class]} ({probs[CLASS_NAMES[predicted_class]]:.1f}%) - {status}")

            except Exception as e:
                print(f"      Error: {e}")
                continue

    # Create summary grid
    print(f"\n{'='*70}")
    print("Creating summary grid...")
    create_summary_grid(all_results, OUTPUT_DIR / 'summary_grid.png')

    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll heatmaps saved to: {OUTPUT_DIR}/")
    print(f"  - Individual heatmaps: {class_key}_1.png, {class_key}_2.png, ...")
    print(f"  - Summary grid: summary_grid.png")
    print("="*70)


if __name__ == '__main__':
    main()
