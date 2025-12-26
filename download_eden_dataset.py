"""
EDEN Garden Dataset Downloader
===============================

Downloads EDEN sample dataset and extracts only the images we need (~31K images).
Final size: ~2-3GB instead of full 11GB.

All downloads to D drive to avoid filling C drive.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import random
import shutil
import ssl

# Configuration
DOWNLOAD_URL = "https://isis-data.science.uva.nl/hale/EDEN-samples/RGB.zip"
DOWNLOAD_DIR = Path(r"D:\Ultimate Dataset\EDEN_temp")
FINAL_DIR = Path(r"D:\Ultimate Dataset\EDEN")
ZIP_PATH = DOWNLOAD_DIR / "RGB.zip"

# How many images we need
TRAIN_IMAGES = 21000  # 70% of 30K training samples
TEST_IMAGES = 10000   # 70% of ~14K validation samples
TOTAL_NEEDED = TRAIN_IMAGES + TEST_IMAGES  # 31,000 images

SEED = 42
random.seed(SEED)


def download_with_progress(url, destination):
    """Download file with progress bar"""
    print(f"\nDownloading from: {url}")
    print(f"Saving to: {destination}")

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100.0 / total_size, 100)
            size_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:.1f}% ({size_mb:.1f} MB / {total_mb:.1f} MB)")
            sys.stdout.flush()

    try:
        # Create SSL context that doesn't verify certificates (for trusted academic sites)
        ssl_context = ssl._create_unverified_context()
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(url, destination, show_progress)
        print("\n  Download complete!")
        return True
    except Exception as e:
        print(f"\n  Error downloading: {e}")
        return False


def extract_limited_images(zip_path, output_dir, max_images):
    """Extract only the first N images from ZIP to save space"""
    print(f"\nExtracting first {max_images:,} images from ZIP...")

    extracted_count = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get all image files from ZIP
        all_files = [f for f in zip_ref.namelist()
                     if any(f.endswith(ext) for ext in image_extensions)]

        print(f"  Found {len(all_files):,} total images in ZIP")

        # Extract only what we need
        for i, file_info in enumerate(all_files):
            if extracted_count >= max_images:
                break

            # Extract this file
            zip_ref.extract(file_info, output_dir)
            extracted_count += 1

            # Progress update every 1000 images
            if (extracted_count % 1000 == 0):
                sys.stdout.write(f"\r  Extracted: {extracted_count:,} / {max_images:,} images")
                sys.stdout.flush()

        print(f"\n  Extraction complete: {extracted_count:,} images extracted")
        return extracted_count


def organize_images(source_dir, train_dir, test_dir, train_count, test_count):
    """Split extracted images into train and test sets"""
    print(f"\nOrganizing images into train ({train_count:,}) and test ({test_count:,})...")

    # Find all images recursively
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    all_images = []
    for ext in image_extensions:
        all_images.extend(source_dir.rglob(ext))

    print(f"  Found {len(all_images):,} images to organize")

    # Shuffle with seed for reproducibility
    random.shuffle(all_images)

    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Split and move
    train_images = all_images[:train_count]
    test_images = all_images[train_count:train_count + test_count]

    print("  Moving images to train folder...")
    for i, img_path in enumerate(train_images):
        dest = train_dir / f"eden_train_{i:05d}{img_path.suffix}"
        shutil.copy2(img_path, dest)
        if (i + 1) % 1000 == 0:
            sys.stdout.write(f"\r    Train: {i+1:,} / {train_count:,}")
            sys.stdout.flush()
    print(f"\n    Train complete: {len(train_images):,} images")

    print("  Moving images to test folder...")
    for i, img_path in enumerate(test_images):
        dest = test_dir / f"eden_test_{i:05d}{img_path.suffix}"
        shutil.copy2(img_path, dest)
        if (i + 1) % 1000 == 0:
            sys.stdout.write(f"\r    Test: {i+1:,} / {test_count:,}")
            sys.stdout.flush()
    print(f"\n    Test complete: {len(test_images):,} images")

    return len(train_images), len(test_images)


def get_directory_size(path):
    """Calculate total size of directory in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


def main():
    print("="*70)
    print("EDEN GARDEN DATASET DOWNLOADER")
    print("="*70)
    print(f"Download URL: {DOWNLOAD_URL}")
    print(f"Temporary download location: {DOWNLOAD_DIR}")
    print(f"Final dataset location: {FINAL_DIR}")
    print(f"Images needed: {TOTAL_NEEDED:,} ({TRAIN_IMAGES:,} train + {TEST_IMAGES:,} test)")
    print("="*70)

    # Check if already exists
    if FINAL_DIR.exists() and (FINAL_DIR / 'train').exists() and (FINAL_DIR / 'test').exists():
        train_count = len(list((FINAL_DIR / 'train').glob('*.jpg'))) + len(list((FINAL_DIR / 'train').glob('*.png')))
        test_count = len(list((FINAL_DIR / 'test').glob('*.jpg'))) + len(list((FINAL_DIR / 'test').glob('*.png')))

        if train_count >= TRAIN_IMAGES and test_count >= TEST_IMAGES:
            print("\n✓ EDEN dataset already exists and has sufficient images!")
            print(f"  Train: {train_count:,} images")
            print(f"  Test: {test_count:,} images")

            response = input("\nDo you want to re-download? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Skipping download. Using existing dataset.")
                return
            else:
                print("Removing existing dataset...")
                shutil.rmtree(FINAL_DIR)

    # Step 1: Create temporary download directory
    print("\nStep 1: Creating download directory...")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {DOWNLOAD_DIR}")

    # Step 2: Check if ZIP already exists, if not download
    if ZIP_PATH.exists():
        print("\nStep 2: ZIP file already exists, skipping download...")
        zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
        print(f"  Using existing ZIP: {ZIP_PATH}")
        print(f"  ZIP file size: {zip_size_mb:.1f} MB")
    else:
        print("\nStep 2: Downloading EDEN sample dataset...")
        print("  This may take 10-30 minutes depending on your connection...")

        if not download_with_progress(DOWNLOAD_URL, ZIP_PATH):
            print("\nError: Download failed!")
            return

        zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
        print(f"  ZIP file size: {zip_size_mb:.1f} MB")

    # Step 3: Extract only what we need
    print(f"\nStep 3: Extracting first {TOTAL_NEEDED:,} images...")
    extract_dir = DOWNLOAD_DIR / "extracted"
    extracted_count = extract_limited_images(ZIP_PATH, extract_dir, TOTAL_NEEDED)

    if extracted_count < TOTAL_NEEDED:
        print(f"\nWarning: Only extracted {extracted_count:,} images (needed {TOTAL_NEEDED:,})")
        print("The sample dataset might be smaller than expected.")
        # Adjust counts proportionally
        ratio = extracted_count / TOTAL_NEEDED
        train_count = int(TRAIN_IMAGES * ratio)
        test_count = extracted_count - train_count
        print(f"Adjusting: {train_count:,} train, {test_count:,} test")
    else:
        train_count = TRAIN_IMAGES
        test_count = TEST_IMAGES

    # Step 4: Organize into train/test
    print("\nStep 4: Organizing into train/test splits...")
    train_dir = FINAL_DIR / "train"
    test_dir = FINAL_DIR / "test"

    final_train, final_test = organize_images(extract_dir, train_dir, test_dir,
                                               train_count, test_count)

    # Step 5: Cleanup
    print("\nStep 5: Cleaning up temporary files...")
    print(f"  Deleting ZIP file ({zip_size_mb:.1f} MB)...")
    ZIP_PATH.unlink()
    print(f"  Deleting extracted temp folder...")
    shutil.rmtree(extract_dir)

    # Remove temp directory if empty
    try:
        DOWNLOAD_DIR.rmdir()
        print(f"  Removed temp directory")
    except:
        pass

    # Step 6: Summary
    print("\n" + "="*70)
    print("DOWNLOAD AND SETUP COMPLETE!")
    print("="*70)

    final_size = get_directory_size(FINAL_DIR)
    print(f"\nFinal dataset location: {FINAL_DIR}")
    print(f"  Train images: {final_train:,} in {FINAL_DIR / 'train'}")
    print(f"  Test images: {final_test:,} in {FINAL_DIR / 'test'}")
    print(f"  Total size: {final_size:.1f} MB (~{final_size/1024:.2f} GB)")
    print(f"\nSpace saved: {zip_size_mb - final_size:.1f} MB by extracting only what we need!")
    print("\nReady to use in training script!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
