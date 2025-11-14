"""
Parse and organize Ultimate Dataset for comprehensive evaluation
Processes BeesAndHornets1 and BeesAndHornets2 datasets
"""

import os
import shutil
from pathlib import Path
import yaml
import random
from collections import defaultdict

def parse_yolo_labels(data_root, data_yaml_path):
    """
    Parse YOLO format labels from BeesAndHornets datasets.

    Returns:
        dict: {'asian_hornets': [image_paths], 'bees': [image_paths]}
    """
    # Read data.yaml to understand class mapping
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Class names: ['Abeille', 'Frelon asiatique']
    # Class 0 = Bee, Class 1 = Asian Hornet
    print(f"Classes found: {config['names']}")

    organized = {'asian_hornets': [], 'bees': []}

    # Process train, val, and test splits
    for split in ['train', 'valid', 'test']:
        images_dir = data_root / split / 'images'
        labels_dir = data_root / split / 'labels'

        if not images_dir.exists():
            print(f"  Skipping {split} (not found)")
            continue

        print(f"  Processing {split}...")

        # Iterate through label files
        for label_file in labels_dir.glob('*.txt'):
            image_name = label_file.stem + '.jpg'
            image_path = images_dir / image_name

            if not image_path.exists():
                # Try .png extension
                image_name = label_file.stem + '.png'
                image_path = images_dir / image_name

            if not image_path.exists():
                continue

            # Read label file (YOLO format: class x y w h)
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])

                if class_id == 0:  # Bee
                    organized['bees'].append(str(image_path))
                elif class_id == 1:  # Asian Hornet
                    organized['asian_hornets'].append(str(image_path))

    return organized


def organize_archive_dataset(archive_root):
    """
    Organize the archive dataset (original 3-class dataset).

    Returns:
        dict: {'asian_hornets': [...], 'european_hornets': [...], 'wasps': [...]}
    """
    organized = {'asian_hornets': [], 'european_hornets': [], 'wasps': []}

    data_root = archive_root / 'data3000' / 'data'

    # Process val split
    for split in ['val']:
        split_root = data_root / split / 'images'

        if not split_root.exists():
            continue

        # Asian hornets (Vespa_velutina)
        asian_dir = split_root / 'Vespa_velutina'
        if asian_dir.exists():
            organized['asian_hornets'].extend([str(p) for p in asian_dir.glob('*.jpg')])

        # European hornets (Vespa_crabro)
        european_dir = split_root / 'Vespa_crabro'
        if european_dir.exists():
            organized['european_hornets'].extend([str(p) for p in european_dir.glob('*.jpg')])

        # Wasps (Vespula_sp)
        wasp_dir = split_root / 'Vespula_sp'
        if wasp_dir.exists():
            organized['wasps'].extend([str(p) for p in wasp_dir.glob('*.jpg')])

    return organized


def create_balanced_test_sets(all_data, output_root, test_sizes):
    """
    Create balanced test sets of different sizes.

    Args:
        all_data: dict with all organized images
        output_root: where to save organized test sets
        test_sizes: dict like {'small': 1000, 'medium': 10000, 'large': 50000}
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("CREATING BALANCED TEST SETS")
    print("="*70)

    for test_name, total_size in test_sizes.items():
        print(f"\nCreating {test_name} test set ({total_size} images)...")

        test_dir = output_root / test_name
        test_dir.mkdir(exist_ok=True)

        # Calculate sizes for binary classification
        # 50% Asian hornets (positive), 50% others (negative)
        asian_hornet_count = total_size // 2
        negative_count = total_size - asian_hornet_count

        # For negatives: distribute among bees, european hornets, wasps
        # Prioritize bees (most abundant), then european hornets, then wasps
        bee_count = int(negative_count * 0.7)  # 70% bees
        european_count = int(negative_count * 0.2)  # 20% european hornets
        wasp_count = negative_count - bee_count - european_count  # 10% wasps

        # Sample images
        sampled = {
            'asian_hornets': random.sample(all_data['asian_hornets'],
                                          min(asian_hornet_count, len(all_data['asian_hornets']))),
            'bees': random.sample(all_data['bees'],
                                 min(bee_count, len(all_data['bees']))),
            'european_hornets': random.sample(all_data['european_hornets'],
                                             min(european_count, len(all_data['european_hornets']))),
            'wasps': random.sample(all_data['wasps'],
                                  min(wasp_count, len(all_data['wasps'])))
        }

        # Copy files to organized structure
        for class_name, image_paths in sampled.items():
            class_dir = test_dir / class_name
            class_dir.mkdir(exist_ok=True)

            for img_path in image_paths:
                src = Path(img_path)
                dst = class_dir / src.name

                # Handle duplicate names
                counter = 1
                while dst.exists():
                    dst = class_dir / f"{src.stem}_{counter}{src.suffix}"
                    counter += 1

                shutil.copy2(src, dst)

            print(f"  {class_name}: {len(image_paths)} images")

        # Save test set statistics
        stats_file = test_dir / 'statistics.txt'
        with open(stats_file, 'w') as f:
            f.write(f"Test Set: {test_name}\n")
            f.write(f"Total images: {total_size}\n\n")
            f.write("Class distribution:\n")
            for class_name, image_paths in sampled.items():
                f.write(f"  {class_name}: {len(image_paths)} ({len(image_paths)/total_size*100:.1f}%)\n")
            f.write("\nBinary classification:\n")
            f.write(f"  POSITIVE (Asian hornets): {len(sampled['asian_hornets'])} ({len(sampled['asian_hornets'])/total_size*100:.1f}%)\n")
            negative_total = len(sampled['bees']) + len(sampled['european_hornets']) + len(sampled['wasps'])
            f.write(f"  NEGATIVE (Others): {negative_total} ({negative_total/total_size*100:.1f}%)\n")


def main():
    print("="*70)
    print("ULTIMATE DATASET PREPARATION")
    print("="*70)

    # Set random seed for reproducibility
    random.seed(42)

    # Paths
    ultimate_root = Path(r"D:\Ultimate Dataset")
    bees_hornets1_root = ultimate_root / "BeesAndHornets1" / "Bee And Asian Hornet Detection"
    bees_hornets2_root = ultimate_root / "BeesAndHornets2" / "Dataset"
    archive_root = ultimate_root / "archive"
    output_root = ultimate_root / "test_organized"

    # Collect all data
    all_data = {
        'asian_hornets': [],
        'bees': [],
        'european_hornets': [],
        'wasps': []
    }

    # Parse BeesAndHornets1
    print("\nProcessing BeesAndHornets1...")
    data1 = parse_yolo_labels(bees_hornets1_root, bees_hornets1_root / 'data.yaml')
    all_data['asian_hornets'].extend(data1['asian_hornets'])
    all_data['bees'].extend(data1['bees'])
    print(f"  Asian hornets: {len(data1['asian_hornets'])}")
    print(f"  Bees: {len(data1['bees'])}")

    # Parse BeesAndHornets2
    print("\nProcessing BeesAndHornets2...")
    data2 = parse_yolo_labels(bees_hornets2_root, bees_hornets2_root / 'data.yaml')
    all_data['asian_hornets'].extend(data2['asian_hornets'])
    all_data['bees'].extend(data2['bees'])
    print(f"  Asian hornets: {len(data2['asian_hornets'])}")
    print(f"  Bees: {len(data2['bees'])}")

    # Parse archive dataset
    print("\nProcessing archive dataset...")
    data3 = organize_archive_dataset(archive_root)
    all_data['asian_hornets'].extend(data3['asian_hornets'])
    all_data['european_hornets'].extend(data3['european_hornets'])
    all_data['wasps'].extend(data3['wasps'])
    print(f"  Asian hornets: {len(data3['asian_hornets'])}")
    print(f"  European hornets: {len(data3['european_hornets'])}")
    print(f"  Wasps: {len(data3['wasps'])}")

    # Print total statistics
    print("\n" + "="*70)
    print("TOTAL DATASET STATISTICS")
    print("="*70)
    print(f"Asian hornets: {len(all_data['asian_hornets'])}")
    print(f"Bees: {len(all_data['bees'])}")
    print(f"European hornets: {len(all_data['european_hornets'])}")
    print(f"Wasps: {len(all_data['wasps'])}")
    print(f"TOTAL: {sum(len(v) for v in all_data.values())} images")

    # Create test sets
    test_sizes = {
        'small': 1000,
        'medium': 10000,
        'large': 50000
    }

    create_balanced_test_sets(all_data, output_root, test_sizes)

    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"Organized datasets saved to: {output_root}")


if __name__ == "__main__":
    main()
