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
MODEL_PATH = "best_binary_hornet_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
TEMPERATURE = 4.0  # Temperature scaling to prevent sigmoid saturation

# Paths to sample images
ASIAN_HORNET_DIR = Path("D:/Ultimate Dataset/BeesAndHornets1/Bee And Asian Hornet Detection/train/images")
BEE_DIR = Path("D:/Ultimate Dataset/BeesAndHornets1/Bee And Asian Hornet Detection/train/images")
EUROPEAN_HORNET_DIR = Path("D:/Ultimate Dataset/european_hornets_gbif")
WASP_DIR = Path("D:/Ultimate Dataset/wasps_gbif")

# Output directory
OUTPUT_DIR = Path("heatmaps")
OUTPUT_DIR.mkdir(exist_ok=True)

class GradCAM:
    """Grad-CAM implementation for visualizing model attention"""

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
        """Generate CAM heatmap"""
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        # Backward pass
        self.model.zero_grad()
        if target_class is None:
            # Use predicted class
            target_class = (torch.sigmoid(output) > 0.5).float()

        output.backward(gradient=target_class)

        # Calculate weights
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted combination
        cam = (weights * activations).sum(dim=0)  # [H, W]

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()

def load_model():
    """Load trained model"""
    print("Loading model...")
    model = models.efficientnet_b3(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    return model

def get_sample_images():
    """Get sample images from each class"""
    samples = {
        'asian_hornet': [],
        'bee': [],
        'european_hornet': [],
        'wasp': []
    }

    # Get Asian hornets (from YOLO labels with class 1) - check both datasets
    label_dirs = [
        (Path("D:/Ultimate Dataset/BeesAndHornets1/Bee And Asian Hornet Detection/train/labels"),
         Path("D:/Ultimate Dataset/BeesAndHornets1/Bee And Asian Hornet Detection/train/images")),
        (Path("D:/Ultimate Dataset/BeesAndHornets2/Dataset/train/labels"),
         Path("D:/Ultimate Dataset/BeesAndHornets2/Dataset/train/images"))
    ]

    for label_dir, img_dir in label_dirs:
        label_files = list(label_dir.glob("*.txt"))[:500]  # Limit to first 500 files
        for label_file in label_files:
            if len(samples['asian_hornet']) >= 5 and len(samples['bee']) >= 5:
                break

            img_name = label_file.stem + ".jpg"
            img_path = img_dir / img_name
            if img_path.exists():
                try:
                    with open(label_file) as f:
                        first_line = f.readline().strip()
                        if first_line:
                            class_id = int(first_line.split()[0])
                            if class_id == 1 and len(samples['asian_hornet']) < 5:
                                samples['asian_hornet'].append(str(img_path))
                            elif class_id == 0 and len(samples['bee']) < 5:
                                samples['bee'].append(str(img_path))
                except:
                    continue

        if len(samples['asian_hornet']) >= 5 and len(samples['bee']) >= 5:
            break

    # Get European hornets
    samples['european_hornet'] = [str(p) for p in list(EUROPEAN_HORNET_DIR.glob("*.jpg"))[:100]]
    samples['european_hornet'] = random.sample(samples['european_hornet'], min(5, len(samples['european_hornet'])))

    # Get wasps
    samples['wasp'] = [str(p) for p in list(WASP_DIR.glob("*.jpg"))[:100]]
    samples['wasp'] = random.sample(samples['wasp'], min(5, len(samples['wasp'])))

    return samples

def preprocess_image(image_path):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    return image, image_tensor

def apply_heatmap(image, heatmap, alpha=0.4):
    """Apply heatmap overlay on image"""
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.width, image.height))

    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Convert PIL image to numpy
    image_np = np.array(image)

    # Overlay
    overlaid = (1 - alpha) * image_np + alpha * heatmap_colored
    overlaid = np.uint8(overlaid)

    return overlaid

def generate_heatmaps_for_samples(model, samples):
    """Generate heatmaps for sample images"""
    print("\nGenerating heatmaps...")

    # Get target layer (last convolutional layer in EfficientNet)
    target_layer = model.features[-1]

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    for category, image_paths in samples.items():
        print(f"\nProcessing {category}...")

        for idx, image_path in enumerate(image_paths[:5]):  # Process up to 5 images per category
            try:
                # Load and preprocess image
                original_image, image_tensor = preprocess_image(image_path)
                image_tensor = image_tensor.to(DEVICE)

                # Get prediction (with temperature scaling)
                with torch.no_grad():
                    output = model(image_tensor)
                    prob = torch.sigmoid(output / TEMPERATURE).item()  # Apply temperature scaling
                    prediction = "Asian Hornet" if prob > 0.5 else "Not Asian Hornet"

                # Generate heatmap
                heatmap = grad_cam.generate_cam(image_tensor)

                # Apply heatmap overlay
                overlaid_image = apply_heatmap(original_image, heatmap)

                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(original_image)
                axes[0].set_title(f'Original Image\n{category.replace("_", " ").title()}')
                axes[0].axis('off')

                # Heatmap
                axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title('Grad-CAM Heatmap')
                axes[1].axis('off')

                # Overlay
                axes[2].imshow(overlaid_image)
                axes[2].set_title(f'Overlay\nPrediction: {prediction}\nConfidence: {prob:.2%}')
                axes[2].axis('off')

                plt.tight_layout()

                # Save figure
                output_path = OUTPUT_DIR / f'{category}_{idx+1}.png'
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"  Saved: {output_path}")
                print(f"    Prediction: {prediction} ({prob:.2%})")

            except Exception as e:
                print(f"  Error processing {image_path}: {e}")
                continue

def create_summary_grid():
    """Create a summary grid of all heatmaps"""
    print("\nCreating summary grid...")

    # Get all generated heatmaps
    heatmap_files = sorted(OUTPUT_DIR.glob("*.png"))

    if not heatmap_files:
        print("No heatmaps found!")
        return

    # Group by category
    categories = {
        'asian_hornet': [],
        'bee': [],
        'european_hornet': [],
        'wasp': []
    }

    for file in heatmap_files:
        for cat in categories.keys():
            if file.stem.startswith(cat):
                categories[cat].append(file)
                break

    # Create grid
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Grad-CAM Heatmap Visualizations - Binary Asian Hornet Classifier',
                 fontsize=16, fontweight='bold')

    row = 0
    for category, files in categories.items():
        if not files:
            continue

        for idx, file in enumerate(files[:5]):
            ax = plt.subplot(4, 5, row * 5 + idx + 1)
            img = plt.imread(file)
            ax.imshow(img)

            if idx == 0:
                ax.set_ylabel(category.replace('_', ' ').title(),
                             fontsize=12, fontweight='bold')

            ax.axis('off')

        row += 1

    plt.tight_layout()
    summary_path = OUTPUT_DIR / 'summary_grid.png'
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Summary grid saved: {summary_path}")

def main():
    print("="*70)
    print("GRAD-CAM HEATMAP GENERATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load model
    model = load_model()

    # Get sample images
    print("\nSelecting sample images...")
    samples = get_sample_images()

    for category, paths in samples.items():
        print(f"  {category}: {len(paths)} images")

    # Generate heatmaps
    generate_heatmaps_for_samples(model, samples)

    # Create summary grid
    create_summary_grid()

    print("\n" + "="*70)
    print("HEATMAP GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated heatmaps saved in: {OUTPUT_DIR}")
    print(f"Total images: {len(list(OUTPUT_DIR.glob('*.png')))}")

if __name__ == "__main__":
    main()
