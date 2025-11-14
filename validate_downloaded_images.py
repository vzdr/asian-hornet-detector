"""
Validate Downloaded GBIF Images
Checks for:
- Corrupted/invalid images
- Duplicate images (by file hash)
- Image sizes and formats
Removes bad images and generates report
"""

import os
import hashlib
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
from collections import defaultdict

def get_file_hash(file_path):
    """Calculate MD5 hash of file for duplicate detection"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def validate_image(image_path):
    """
    Check if image is valid and can be opened
    Returns: (is_valid, error_message, width, height, format)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            img_format = img.format

            # Check minimum size (too small = not useful)
            if width < 50 or height < 50:
                return False, "Image too small", width, height, img_format

            # Verify image can be loaded (catches some corruption)
            img.verify()

            return True, None, width, height, img_format

    except Exception as e:
        return False, str(e), 0, 0, None


def validate_directory(directory):
    """
    Validate all images in a directory
    Returns: dict with statistics and lists of bad/duplicate files
    """
    directory = Path(directory)
    print(f"\nValidating: {directory}")
    print("=" * 70)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    all_files = []

    for ext in image_extensions:
        all_files.extend(directory.glob(f"*{ext}"))
        all_files.extend(directory.glob(f"*{ext.upper()}"))

    total_files = len(all_files)
    print(f"Found {total_files} image files")

    if total_files == 0:
        return {
            'total_files': 0,
            'valid_files': 0,
            'corrupted_files': 0,
            'duplicates': 0,
            'total_size_mb': 0
        }

    # Track results
    valid_files = []
    corrupted_files = []
    file_hashes = defaultdict(list)
    total_size = 0
    image_stats = {
        'formats': defaultdict(int),
        'sizes': [],
        'dimensions': []
    }

    print("\nChecking images...")
    for i, file_path in enumerate(all_files, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{total_files} images...")

        # Get file size
        file_size = file_path.stat().st_size
        total_size += file_size

        # Validate image
        is_valid, error, width, height, img_format = validate_image(file_path)

        if is_valid:
            valid_files.append(file_path)

            # Calculate hash for duplicate detection
            file_hash = get_file_hash(file_path)
            file_hashes[file_hash].append(file_path)

            # Collect stats
            image_stats['formats'][img_format] += 1
            image_stats['sizes'].append(file_size / 1024)  # KB
            image_stats['dimensions'].append((width, height))
        else:
            corrupted_files.append((file_path, error))
            print(f"  CORRUPTED: {file_path.name} - {error}")

    # Find duplicates
    duplicates = {hash_val: files for hash_val, files in file_hashes.items() if len(files) > 1}

    print(f"\nValidation complete!")
    print(f"  Valid images: {len(valid_files)}")
    print(f"  Corrupted images: {len(corrupted_files)}")
    print(f"  Duplicate groups: {len(duplicates)}")
    print(f"  Total size: {total_size / (1024*1024):.2f} MB")

    return {
        'total_files': total_files,
        'valid_files': len(valid_files),
        'corrupted_files': len(corrupted_files),
        'duplicates': len(duplicates),
        'total_size_mb': total_size / (1024*1024),
        'corrupted_list': corrupted_files,
        'duplicate_dict': duplicates,
        'image_stats': image_stats
    }


def remove_bad_images(validation_results, directory, remove_corrupted=True, remove_duplicates=True):
    """
    Remove corrupted images and duplicates

    Args:
        validation_results: Results from validate_directory()
        directory: Path to directory
        remove_corrupted: Whether to delete corrupted images
        remove_duplicates: Whether to delete duplicate images (keeps first)
    """
    directory = Path(directory)
    removed_count = 0

    print(f"\nCleaning up: {directory}")
    print("=" * 70)

    # Remove corrupted images
    if remove_corrupted and validation_results['corrupted_files'] > 0:
        print(f"\nRemoving {validation_results['corrupted_files']} corrupted images...")
        for file_path, error in validation_results['corrupted_list']:
            try:
                file_path.unlink()
                removed_count += 1
                print(f"  Removed: {file_path.name}")
            except Exception as e:
                print(f"  ERROR removing {file_path.name}: {e}")

    # Remove duplicates (keep first, remove rest)
    if remove_duplicates and validation_results['duplicates'] > 0:
        print(f"\nRemoving duplicates (keeping first of each group)...")
        duplicate_count = 0
        for hash_val, files in validation_results['duplicate_dict'].items():
            # Keep first file, remove rest
            for file_path in files[1:]:
                try:
                    file_path.unlink()
                    removed_count += 1
                    duplicate_count += 1
                    print(f"  Removed duplicate: {file_path.name}")
                except Exception as e:
                    print(f"  ERROR removing {file_path.name}: {e}")
        print(f"  Removed {duplicate_count} duplicate images")

    print(f"\nTotal files removed: {removed_count}")
    return removed_count


def generate_report(european_results, wasp_results, output_file):
    """Generate comprehensive validation report"""

    report = {
        'validation_time': datetime.now().isoformat(),
        'european_hornets': {
            'total_files': european_results['total_files'],
            'valid_files': european_results['valid_files'],
            'corrupted_files': european_results['corrupted_files'],
            'duplicates': european_results['duplicates'],
            'total_size_mb': european_results['total_size_mb']
        },
        'wasps': {
            'total_files': wasp_results['total_files'],
            'valid_files': wasp_results['valid_files'],
            'corrupted_files': wasp_results['corrupted_files'],
            'duplicates': wasp_results['duplicates'],
            'total_size_mb': wasp_results['total_size_mb']
        },
        'summary': {
            'total_valid_images': european_results['valid_files'] + wasp_results['valid_files'],
            'total_size_mb': european_results['total_size_mb'] + wasp_results['total_size_mb']
        }
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def main():
    """Main validation function"""

    print("=" * 70)
    print("GBIF IMAGE VALIDATION")
    print("=" * 70)

    base_dir = Path("D:/Ultimate Dataset")
    european_dir = base_dir / "european_hornets_gbif"
    wasps_dir = base_dir / "wasps_gbif"

    # Check directories exist
    if not european_dir.exists():
        print(f"ERROR: Directory not found: {european_dir}")
        return

    if not wasps_dir.exists():
        print(f"ERROR: Directory not found: {wasps_dir}")
        return

    # Validate European Hornets
    print("\n" + "=" * 70)
    print("PHASE 1: VALIDATE EUROPEAN HORNETS")
    print("=" * 70)
    european_results = validate_directory(european_dir)

    # Validate Wasps
    print("\n" + "=" * 70)
    print("PHASE 2: VALIDATE WASPS")
    print("=" * 70)
    wasp_results = validate_directory(wasps_dir)

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)

    report_file = base_dir / "gbif_validation_report.json"
    report = generate_report(european_results, wasp_results, report_file)
    print(f"\nReport saved to: {report_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nEuropean Hornets:")
    print(f"  Total files: {european_results['total_files']}")
    print(f"  Valid: {european_results['valid_files']}")
    print(f"  Corrupted: {european_results['corrupted_files']}")
    print(f"  Duplicates: {european_results['duplicates']}")
    print(f"  Size: {european_results['total_size_mb']:.2f} MB")

    print(f"\nWasps:")
    print(f"  Total files: {wasp_results['total_files']}")
    print(f"  Valid: {wasp_results['valid_files']}")
    print(f"  Corrupted: {wasp_results['corrupted_files']}")
    print(f"  Duplicates: {wasp_results['duplicates']}")
    print(f"  Size: {wasp_results['total_size_mb']:.2f} MB")

    print(f"\nTotal Valid Images: {report['summary']['total_valid_images']}")
    print(f"Total Size: {report['summary']['total_size_mb']:.2f} MB")

    # Ask user if they want to clean up
    print("\n" + "=" * 70)
    print("CLEANUP OPTIONS")
    print("=" * 70)
    print("\nWould you like to remove corrupted and duplicate images?")
    print("This will delete files permanently.")
    print()
    response = input("Remove bad images? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        print("\nStarting cleanup...")

        # Clean European Hornets
        print("\n" + "=" * 70)
        print("CLEANING: EUROPEAN HORNETS")
        print("=" * 70)
        removed_eu = remove_bad_images(european_results, european_dir, remove_corrupted=True, remove_duplicates=True)

        # Clean Wasps
        print("\n" + "=" * 70)
        print("CLEANING: WASPS")
        print("=" * 70)
        removed_wasps = remove_bad_images(wasp_results, wasps_dir, remove_corrupted=True, remove_duplicates=True)

        print("\n" + "=" * 70)
        print("CLEANUP COMPLETE")
        print("=" * 70)
        print(f"\nTotal files removed: {removed_eu + removed_wasps}")
        print(f"  European Hornets: {removed_eu}")
        print(f"  Wasps: {removed_wasps}")

        # Re-validate to get final counts
        print("\nRe-validating directories...")
        final_eu = validate_directory(european_dir)
        final_wasps = validate_directory(wasps_dir)

        print("\n" + "=" * 70)
        print("FINAL COUNTS")
        print("=" * 70)
        print(f"\nEuropean Hornets: {final_eu['valid_files']} images")
        print(f"Wasps: {final_wasps['valid_files']} images")
        print(f"Total: {final_eu['valid_files'] + final_wasps['valid_files']} images")

    else:
        print("\nSkipping cleanup. Bad images remain in directories.")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
