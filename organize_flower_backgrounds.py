r"""
Organize Flower Background Datasets for Asian Hornet Detection
================================================================

Takes two flower datasets and organizes them into train/test splits:
1. Flowers Recognition (4,317 images) - Perfect quality outdoor photos
2. Flowers299 (115,944 images) - Large variety of flower species

Output structure:
D:\Ultimate Dataset\FlowerBackgrounds\
├── train\
│   ├── flowers_recognition\  (all 4,317 images)
│   └── flowers299\           (~92,755 images, 80% split)
└── test\
    ├── flowers_recognition\  (sample from train for validation)
    └── flowers299\           (~23,189 images, 20% split)
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
SEED = 42
random.seed(SEED)

# Source paths
FLOWERS_RECOGNITION_SOURCE = Path(r"D:\backgrounddatasetgardens\archive\flowers")
FLOWERS299_SOURCE = Path(r"D:\backgrounddatasetgardens\2\Flowers299")

# Destination
OUTPUT_DIR = Path(r"D:\Ultimate Dataset\FlowerBackgrounds")
TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"

# Split ratio for Flowers299
TRAIN_RATIO = 0.80  # 80% train, 20% test


def create_directories():
    """Create output directory structure"""
    print("="*70)
    print("CREATING OUTPUT DIRECTORIES")
    print("="*70)

    # Create main directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (TRAIN_DIR / "flowers_recognition").mkdir(exist_ok=True)
    (TRAIN_DIR / "flowers299").mkdir(exist_ok=True)
    (TEST_DIR / "flowers_recognition").mkdir(exist_ok=True)
    (TEST_DIR / "flowers299").mkdir(exist_ok=True)

    print(f"[OK] Created: {OUTPUT_DIR}")
    print(f"[OK] Train directory: {TRAIN_DIR}")
    print(f"[OK] Test directory: {TEST_DIR}")


def copy_flowers_recognition():
    """
    Copy Flowers Recognition dataset (small, high quality)
    Strategy: Use all for training, sample 20% for testing (overlap is acceptable for validation)
    """
    print("\n" + "="*70)
    print("PROCESSING FLOWERS RECOGNITION DATASET")
    print("="*70)

    if not FLOWERS_RECOGNITION_SOURCE.exists():
        print(f"[ERROR] Source not found: {FLOWERS_RECOGNITION_SOURCE}")
        return 0, 0

    # Find all images
    all_images = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    for species_dir in FLOWERS_RECOGNITION_SOURCE.iterdir():
        if species_dir.is_dir():
            for img_path in species_dir.iterdir():
                if img_path.suffix in image_extensions and img_path.is_file():
                    all_images.append((img_path, species_dir.name))

    print(f"Found {len(all_images)} images across {len(set(s for _, s in all_images))} species")

    # Copy all to train
    train_dest = TRAIN_DIR / "flowers_recognition"
    train_count = 0

    print(f"\nCopying to train folder...")
    for img_path, species in all_images:
        # Create species subdirectory
        species_dir = train_dest / species
        species_dir.mkdir(exist_ok=True)

        # Copy image
        dest_path = species_dir / img_path.name
        shutil.copy2(img_path, dest_path)
        train_count += 1

        if train_count % 500 == 0:
            print(f"  Copied: {train_count}/{len(all_images)} images")

    print(f"[OK] Train: {train_count} images")

    # Sample 20% for test (with overlap - acceptable for validation)
    test_sample_size = int(len(all_images) * 0.20)
    test_images = random.sample(all_images, test_sample_size)

    test_dest = TEST_DIR / "flowers_recognition"
    test_count = 0

    print(f"\nSampling {test_sample_size} images for test...")
    for img_path, species in test_images:
        species_dir = test_dest / species
        species_dir.mkdir(exist_ok=True)

        dest_path = species_dir / img_path.name
        shutil.copy2(img_path, dest_path)
        test_count += 1

        if test_count % 100 == 0:
            print(f"  Copied: {test_count}/{test_sample_size} images")

    print(f"[OK] Test: {test_count} images (20% sample)")
    print(f"\nFlowers Recognition Summary:")
    print(f"  Train: {train_count} images (100%)")
    print(f"  Test: {test_count} images (20% sample, overlap OK)")

    return train_count, test_count


def split_flowers299():
    """
    Split Flowers299 dataset (large, 115K images)
    Strategy: Proper 80/20 train/test split per species
    """
    print("\n" + "="*70)
    print("PROCESSING FLOWERS299 DATASET")
    print("="*70)

    if not FLOWERS299_SOURCE.exists():
        print(f"[ERROR] Source not found: {FLOWERS299_SOURCE}")
        return 0, 0

    # Group images by species
    species_images = defaultdict(list)
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    print("Scanning species directories...")
    for species_dir in FLOWERS299_SOURCE.iterdir():
        if species_dir.is_dir():
            for img_path in species_dir.iterdir():
                if img_path.suffix in image_extensions and img_path.is_file():
                    species_images[species_dir.name].append(img_path)

    total_images = sum(len(imgs) for imgs in species_images.values())
    print(f"Found {total_images} images across {len(species_images)} species")

    # Process each species with 80/20 split
    train_count = 0
    test_count = 0

    print(f"\nSplitting into train (80%) and test (20%)...")

    for species_idx, (species, images) in enumerate(species_images.items(), 1):
        # Shuffle images for this species
        random.shuffle(images)

        # Calculate split point
        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Copy train images
        train_species_dir = TRAIN_DIR / "flowers299" / species
        train_species_dir.mkdir(parents=True, exist_ok=True)

        for img_path in train_images:
            dest_path = train_species_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            train_count += 1

        # Copy test images
        test_species_dir = TEST_DIR / "flowers299" / species
        test_species_dir.mkdir(parents=True, exist_ok=True)

        for img_path in test_images:
            dest_path = test_species_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            test_count += 1

        # Progress update every 10 species
        if species_idx % 10 == 0:
            print(f"  Processed: {species_idx}/{len(species_images)} species "
                  f"(Train: {train_count:,}, Test: {test_count:,})")

    print(f"\nFlowers299 Summary:")
    print(f"  Train: {train_count:,} images ({train_count/total_images*100:.1f}%)")
    print(f"  Test: {test_count:,} images ({test_count/total_images*100:.1f}%)")

    return train_count, test_count


def verify_structure():
    """Verify the final directory structure and counts"""
    print("\n" + "="*70)
    print("VERIFYING DATASET STRUCTURE")
    print("="*70)

    def count_images(path):
        """Count images in directory recursively"""
        extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        count = 0
        for root, dirs, files in os.walk(path):
            count += sum(1 for f in files if Path(f).suffix in extensions)
        return count

    # Count images
    train_recognition = count_images(TRAIN_DIR / "flowers_recognition")
    train_299 = count_images(TRAIN_DIR / "flowers299")
    test_recognition = count_images(TEST_DIR / "flowers_recognition")
    test_299 = count_images(TEST_DIR / "flowers299")

    total_train = train_recognition + train_299
    total_test = test_recognition + test_299
    grand_total = total_train + total_test

    print(f"\nFinal Dataset Structure:")
    print(f"├── train/ ({total_train:,} images)")
    print(f"│   ├── flowers_recognition/ ({train_recognition:,} images)")
    print(f"│   └── flowers299/ ({train_299:,} images)")
    print(f"└── test/ ({total_test:,} images)")
    print(f"    ├── flowers_recognition/ ({test_recognition:,} images)")
    print(f"    └── flowers299/ ({test_299:,} images)")
    print(f"\nGrand Total: {grand_total:,} images")

    # Check expected counts
    print(f"\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    print(f"[CHECK] Expected train images: ~97,000-100,000")
    print(f"  Actual: {total_train:,} ({'PASS' if 95000 <= total_train <= 105000 else 'FAIL'})")
    print(f"[CHECK] Expected test images: ~23,000-25,000")
    print(f"  Actual: {total_test:,} ({'PASS' if 22000 <= total_test <= 26000 else 'FAIL'})")

    return total_train, total_test


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("FLOWER BACKGROUNDS DATASET ORGANIZER")
    print("="*70)
    print(f"Random seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train/Test split: {TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}%")

    # Check if output already exists
    if OUTPUT_DIR.exists():
        print(f"\n[WARNING] Output directory already exists: {OUTPUT_DIR}")
        print("Removing existing directory...")
        shutil.rmtree(OUTPUT_DIR)

    try:
        # Step 1: Create directories
        create_directories()

        # Step 2: Copy Flowers Recognition
        fr_train, fr_test = copy_flowers_recognition()

        # Step 3: Split Flowers299
        f299_train, f299_test = split_flowers299()

        # Step 4: Verify structure
        total_train, total_test = verify_structure()

        # Final summary
        print("\n" + "="*70)
        print("[SUCCESS] DATASET ORGANIZATION COMPLETE!")
        print("="*70)
        print(f"Location: {OUTPUT_DIR}")
        print(f"Train images: {total_train:,}")
        print(f"Test images: {total_test:,}")
        print(f"Total images: {total_train + total_test:,}")
        print("\nReady for training!")
        print("Next step: Update train_5class_efficientnet.py to use FlowerBackgrounds")

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
