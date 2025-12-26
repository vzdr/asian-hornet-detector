"""
5-Class Model Grad-CAM Heatmap Visualization
=============================================

Generates Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations
to show which regions of images the model focuses on for classification.

Also creates:
- Per-class performance heatmaps
- Confidence distribution visualizations
- Sample predictions with activation maps
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import pickle
import cv2
from sklearn.metrics import classification_report
import seaborn as sns

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Configuration
NUM_CLASSES = 5
CLASS_NAMES = ['Asian Hornet', 'Bee', 'European Hornet', 'Wasp', 'Random Images']
MODEL_PATH = 'multiclass_models/best_5class_model.pth'
OUTPUT_DIR = '5class_betterbackground'
SAMPLES_PER_CLASS = 10  # Number of sample images to visualize per class

# Data paths
DATA_SOURCES = {
    'bees_hornets1': r'D:\Ultimate Dataset\BeesAndHornets1\Bee And Asian Hornet Detection',
    'bees_hornets2': r'D:\Ultimate Dataset\BeesAndHornets2\Dataset',
    'gbif_european_hornets': r'D:\Ultimate Dataset\european_hornets_gbif',
    'gbif_wasps': r'D:\Ultimate Dataset\wasps_gbif',
    'cifar10': r'D:\Ultimate Dataset\cifar-10-python\cifar-10-batches-py',
}


class GradCAM:
    """Grad-CAM implementation for EfficientNet"""

    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class):
        """Generate CAM for target class"""
        # Ensure input is on correct device
        input_image = input_image.to(self.device)

        # Forward pass
        output = self.model(input_image)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        class_loss = output[:, target_class]
        class_loss.backward()

        # Get gradients and activations (ensure they're on the same device)
        gradients = self.gradients[0].to(self.device)  # (C, H, W)
        activations = self.activations[0].to(self.device)  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


def load_model():
    """Load trained 5-class model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model architecture
    model = models.efficientnet_b3(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )

    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Handle both dictionary and direct state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {MODEL_PATH}")
    return model, device


def get_sample_images():
    """Get sample images from each class"""
    samples = {i: [] for i in range(NUM_CLASSES)}

    # Asian Hornets (class 0) and Bees (class 1) - YOLO format
    # These datasets have images and labels folders, labels contain class ID
    yolo_data_dir = Path(DATA_SOURCES['bees_hornets1']) / 'train'
    images_dir = yolo_data_dir / 'images'
    labels_dir = yolo_data_dir / 'labels'

    if images_dir.exists() and labels_dir.exists():
        asian_count = 0
        bee_count = 0

        for label_file in labels_dir.glob('*.txt'):
            if asian_count >= SAMPLES_PER_CLASS and bee_count >= SAMPLES_PER_CLASS:
                break

            # Read label file to get class
            try:
                with open(label_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        class_id = int(first_line.split()[0])

                        # Get corresponding image
                        img_file = images_dir / (label_file.stem + '.jpg')
                        if not img_file.exists():
                            img_file = images_dir / (label_file.stem + '.png')

                        if img_file.exists():
                            if class_id == 0 and asian_count < SAMPLES_PER_CLASS:  # Asian Hornet
                                samples[0].append(img_file)
                                asian_count += 1
                            elif class_id == 1 and bee_count < SAMPLES_PER_CLASS:  # Bee
                                samples[1].append(img_file)
                                bee_count += 1
            except:
                continue

    # European Hornets (class 2)
    euro_dir = Path(DATA_SOURCES['gbif_european_hornets'])
    if euro_dir.exists():
        images = list(euro_dir.glob('*.jpg'))[:SAMPLES_PER_CLASS]
        samples[2] = images

    # Wasps (class 3)
    wasp_dir = Path(DATA_SOURCES['gbif_wasps'])
    if wasp_dir.exists():
        images = list(wasp_dir.glob('*.jpg'))[:SAMPLES_PER_CLASS]
        samples[3] = images

    # Random Images (class 4) - from real flower datasets
    all_flower_images = []

    # Source 1: Flowers299 dataset
    flowers299_dir = Path(r'D:\backgrounddatasetgardens\2\Flowers299')
    if flowers299_dir.exists():
        for subdir in flowers299_dir.iterdir():
            if subdir.is_dir():
                imgs = list(subdir.glob('*.jpg'))[:2] + list(subdir.glob('*.png'))[:2]
                all_flower_images.extend(imgs)
            if len(all_flower_images) >= SAMPLES_PER_CLASS:
                break

    # Source 2: archive/flowers dataset
    archive_flowers_dir = Path(r'D:\backgrounddatasetgardens\archive\flowers')
    if archive_flowers_dir.exists() and len(all_flower_images) < SAMPLES_PER_CLASS:
        for subdir in archive_flowers_dir.iterdir():
            if subdir.is_dir():
                imgs = list(subdir.glob('*.jpg'))[:2] + list(subdir.glob('*.png'))[:2]
                all_flower_images.extend(imgs)
            if len(all_flower_images) >= SAMPLES_PER_CLASS:
                break

    samples[4] = all_flower_images[:SAMPLES_PER_CLASS]

    if not samples[4]:
        print("WARNING: No flower images found, falling back to CIFAR-10")
        cifar_path = Path(DATA_SOURCES['cifar10']) / 'test_batch'
        if cifar_path.exists():
            with open(cifar_path, 'rb') as f:
                data_dict = pickle.load(f, encoding='bytes')
            images_array = data_dict[b'data'][:SAMPLES_PER_CLASS]
            images_array = images_array.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            cifar_images = [Image.fromarray(img) for img in images_array]
            samples[4] = cifar_images

    return samples


def apply_colormap_on_image(org_im, activation, colormap_name='jet'):
    """Apply colormap on heatmap and overlay on original image"""
    # Resize activation to match image size
    height, width = org_im.shape[:2]
    activation_resized = cv2.resize(activation, (width, height))

    # Apply colormap
    colormap = cm.get_cmap(colormap_name)
    heatmap = colormap(activation_resized)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)

    # Overlay
    org_im = org_im.astype(float)
    heatmap = heatmap.astype(float)
    overlayed = 0.6 * org_im + 0.4 * heatmap
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)

    return overlayed, heatmap


def visualize_gradcam(model, device, samples, output_dir):
    """Generate and save Grad-CAM visualizations"""
    print("\n" + "="*70)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*70)

    # Get target layer (last conv layer in EfficientNet)
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer, device)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Process each class
    for class_idx in range(NUM_CLASSES):
        class_name = CLASS_NAMES[class_idx]
        print(f"\nProcessing {class_name}...")

        class_samples = samples[class_idx]
        if not class_samples:
            print(f"  No samples found for {class_name}")
            continue

        # Create class subdirectory
        class_dir = Path(output_dir) / f"class_{class_idx}_{class_name.replace(' ', '_')}"
        class_dir.mkdir(parents=True, exist_ok=True)

        for idx, img_source in enumerate(class_samples):
            try:
                # Load image
                if isinstance(img_source, (str, Path)):
                    img_pil = Image.open(img_source).convert('RGB')
                else:
                    img_pil = img_source.convert('RGB')

                # Prepare for model
                img_tensor = transform(img_pil).unsqueeze(0).to(device)

                # Get prediction
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = output.argmax(dim=1).item()
                    confidence = probabilities[0][predicted_class].item()

                # Generate CAM for predicted class
                cam = gradcam.generate_cam(img_tensor, predicted_class)

                # Convert image for visualization
                img_np = np.array(img_pil.resize((224, 224)))

                # Apply heatmap
                overlayed, heatmap = apply_colormap_on_image(img_np, cam)

                # Create visualization
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                # Original image
                axes[0].imshow(img_np)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                # Heatmap
                axes[1].imshow(heatmap)
                axes[1].set_title('Activation Heatmap')
                axes[1].axis('off')

                # Overlay
                axes[2].imshow(overlayed)
                axes[2].set_title('Grad-CAM Overlay')
                axes[2].axis('off')

                # Prediction confidence
                axes[3].barh(range(NUM_CLASSES), probabilities[0].cpu().numpy())
                axes[3].set_yticks(range(NUM_CLASSES))
                axes[3].set_yticklabels(CLASS_NAMES, fontsize=8)
                axes[3].set_xlabel('Confidence')
                axes[3].set_title(f'Predicted: {CLASS_NAMES[predicted_class]}\n({confidence:.2%})')
                axes[3].set_xlim([0, 1])

                # Add grid
                axes[3].grid(axis='x', alpha=0.3)

                plt.tight_layout()

                # Save
                save_path = class_dir / f"sample_{idx+1}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"  Saved: {save_path.name} (Predicted: {CLASS_NAMES[predicted_class]}, {confidence:.2%})")

            except Exception as e:
                print(f"  Error processing sample {idx+1}: {e}")
                continue

    print("\nGrad-CAM visualizations complete!")


def create_performance_heatmap(output_dir):
    """Create heatmap showing per-class performance metrics"""
    print("\n" + "="*70)
    print("GENERATING PERFORMANCE METRICS HEATMAP")
    print("="*70)

    # Read classification report from evaluation
    # For now, create a placeholder - in real use, load from evaluation results
    # You would load actual metrics from the evaluation script output

    # Example metrics (replace with actual values from evaluation)
    metrics = {
        'Precision': [0.89, 0.91, 0.85, 0.87, 0.95],
        'Recall': [0.88, 0.90, 0.87, 0.86, 0.94],
        'F1-Score': [0.88, 0.90, 0.86, 0.86, 0.94],
        'Support': [3070, 4180, 36811, 30367, 10000]
    }

    # Create dataframe-like structure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for heatmap (exclude support from color mapping)
    data = np.array([metrics['Precision'], metrics['Recall'], metrics['F1-Score']])

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)

    # Set ticks and labels
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(['Precision', 'Recall', 'F1-Score'])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=15)

    # Annotate cells with values
    for i in range(3):
        for j in range(NUM_CLASSES):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')

    # Add support counts as text below
    support_text = '\n'.join([f"{CLASS_NAMES[i]}: {metrics['Support'][i]:,} samples"
                              for i in range(NUM_CLASSES)])

    plt.title('Per-Class Performance Metrics\n5-Class Asian Hornet Classifier',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    save_path = Path(output_dir) / 'performance_metrics_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Performance heatmap saved: {save_path}")


def create_confidence_distribution(output_dir):
    """Create visualization of prediction confidence distributions"""
    print("\n" + "="*70)
    print("GENERATING CONFIDENCE DISTRIBUTION PLOT")
    print("="*70)

    # This would typically load actual confidence scores from evaluation
    # For now, create representative distributions

    fig, ax = plt.subplots(figsize=(12, 6))

    # Simulate confidence distributions (replace with actual data)
    np.random.seed(42)
    confidences = {
        'Asian Hornet': np.random.beta(8, 2, 3070),
        'Bee': np.random.beta(9, 2, 4180),
        'European Hornet': np.random.beta(7, 2, 36811),
        'Wasp': np.random.beta(7, 2, 30367),
        'Random Images': np.random.beta(10, 1, 10000)
    }

    # Create violin plot
    positions = range(NUM_CLASSES)
    parts = ax.violinplot([confidences[name] for name in CLASS_NAMES],
                          positions=positions, showmeans=True, showmedians=True)

    # Customize
    ax.set_xticks(positions)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylabel('Prediction Confidence')
    ax.set_ylim([0, 1.05])
    ax.set_title('Prediction Confidence Distribution by Class\n5-Class Asian Hornet Classifier',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add mean confidence labels
    for i, name in enumerate(CLASS_NAMES):
        mean_conf = confidences[name].mean()
        ax.text(i, 1.02, f'{mean_conf:.2%}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save
    save_path = Path(output_dir) / 'confidence_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confidence distribution plot saved: {save_path}")


def main():
    print("="*70)
    print("5-CLASS MODEL GRAD-CAM HEATMAP GENERATION")
    print("="*70)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Load model
    model, device = load_model()

    # Get sample images
    print("\nCollecting sample images...")
    samples = get_sample_images()
    for i, class_name in enumerate(CLASS_NAMES):
        count = len(samples[i]) if isinstance(samples[i], list) else 0
        print(f"  {class_name}: {count} samples")

    # Generate Grad-CAM visualizations
    visualize_gradcam(model, device, samples, OUTPUT_DIR)

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nOutput saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - Grad-CAM visualizations for each class (in subdirectories)")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
