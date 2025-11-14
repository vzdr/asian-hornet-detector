"""
GBIF Image Downloader V2 - With Improved Error Handling
Downloads European Hornet and Wasp images from GBIF using direct URL downloads
Features:
- Proper timeout handling
- Resume capability
- Progress tracking
- Concurrent downloads with connection pooling
"""

import os
from pathlib import Path
from datetime import datetime
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import hashlib

# Import gbif-dl only for API queries
from gbif_dl import api

# Progress tracking
progress_lock = Lock()
stats = {
    'success': 0,
    'failed': 0,
    'skipped': 0,
    'timeouts': 0
}


def setup_directories():
    """Create directory structure for downloads"""
    base_dir = Path("D:/Ultimate Dataset")

    european_hornets_dir = base_dir / "european_hornets_gbif"
    wasps_dir = base_dir / "wasps_gbif"

    european_hornets_dir.mkdir(parents=True, exist_ok=True)
    wasps_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GBIF IMAGE DOWNLOAD V2 - WITH IMPROVED ERROR HANDLING")
    print("=" * 70)
    print(f"\nDirectories:")
    print(f"  European Hornets: {european_hornets_dir}")
    print(f"  Wasps: {wasps_dir}")
    print()

    return european_hornets_dir, wasps_dir


def download_single_image(media_data, output_dir, timeout=10):
    """
    Download a single image with timeout and error handling

    Args:
        media_data: MediaData dictionary from gbif-dl
        output_dir: Where to save the image
        timeout: Download timeout in seconds

    Returns:
        (success, message)
    """
    try:
        # Get the image URL from dictionary
        url = media_data['url']

        # Generate filename from URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

        # Try to get file extension from URL
        ext = url.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            ext = 'jpg'

        # Get GBIF ID from label dictionary
        gbif_id = media_data['label'].get('gbifID', url_hash)
        filename = f"{gbif_id}_{url_hash}.{ext}"
        filepath = output_dir / filename

        # Skip if already exists
        if filepath.exists() and filepath.stat().st_size > 0:
            return True, "skipped"

        # Download with timeout
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        # Save image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Verify file was written
        if filepath.stat().st_size > 0:
            return True, "success"
        else:
            filepath.unlink()  # Remove empty file
            return False, "empty_file"

    except requests.exceptions.Timeout:
        return False, "timeout"
    except requests.exceptions.RequestException as e:
        return False, f"request_error: {type(e).__name__}"
    except Exception as e:
        return False, f"error: {type(e).__name__}"


def download_batch(media_data_list, output_dir, max_workers=20, timeout=10):
    """
    Download a batch of images using thread pool

    Args:
        media_data_list: List of MediaData objects
        output_dir: Where to save images
        max_workers: Number of concurrent downloads
        timeout: Timeout per download in seconds
    """
    total = len(media_data_list)

    print(f"\nDownloading {total} images...")
    print(f"  Workers: {max_workers}")
    print(f"  Timeout: {timeout}s per image")
    print()

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_single_image, media_data, output_dir, timeout): i
            for i, media_data in enumerate(media_data_list)
        }

        # Process completed downloads
        for future in as_completed(future_to_url):
            success, message = future.result()

            with progress_lock:
                if message == "skipped":
                    stats['skipped'] += 1
                elif message == "success":
                    stats['success'] += 1
                elif message == "timeout":
                    stats['timeouts'] += 1
                    stats['failed'] += 1
                else:
                    stats['failed'] += 1

                # Print progress every 100 images
                completed = stats['success'] + stats['failed'] + stats['skipped']
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Progress: {completed}/{total} "
                          f"({rate:.1f} images/s) - "
                          f"Success: {stats['success']}, Failed: {stats['failed']}, "
                          f"Timeouts: {stats['timeouts']}, Skipped: {stats['skipped']}")

    elapsed = time.time() - start_time
    print(f"\n  Download completed in {elapsed:.1f}s ({total/elapsed:.1f} images/s)")


def download_species(taxon_key, species_name, output_dir, max_images=50000, batch_size=1000):
    """
    Download images for a specific species from GBIF

    Args:
        taxon_key: GBIF taxon key
        species_name: Name for logging
        output_dir: Where to save images
        max_images: Maximum number of images to download
        batch_size: Process in batches to avoid memory issues
    """
    print("=" * 70)
    print(f"DOWNLOADING: {species_name}")
    print("=" * 70)
    print(f"Taxon Key: {taxon_key}")
    print(f"Output Directory: {output_dir}")
    print(f"Max Images: {max_images}")
    print()

    # Reset stats
    global stats
    stats = {'success': 0, 'failed': 0, 'skipped': 0, 'timeouts': 0}

    log_file = output_dir / "download_log.txt"

    try:
        # Generate MediaData objects from GBIF API
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Querying GBIF API for image URLs...")

        media_generator = api.gbif_query_generator(
            taxon_key=str(taxon_key),
            media_type='StillImage'
        )

        # Collect MediaData objects
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Collecting media data...")
        media_data_list = []

        for media_data in media_generator:
            media_data_list.append(media_data)
            if len(media_data_list) >= max_images:
                break
            if len(media_data_list) % 1000 == 0:
                print(f"  Collected {len(media_data_list)} media items...")

        total_urls = len(media_data_list)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {total_urls} image URLs")

        if total_urls == 0:
            print(f"WARNING: No images found for {species_name}")
            with open(log_file, 'w') as f:
                f.write(f"No images found for {species_name} (taxon_key={taxon_key})\n")
                f.write(f"Query time: {datetime.now()}\n")
            return 0

        # Download in batches
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting downloads...")
        print(f"  Processing in batches of {batch_size}")
        print()

        overall_start = time.time()

        for i in range(0, total_urls, batch_size):
            batch = media_data_list[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (total_urls + batch_size - 1) // batch_size

            print(f"\n{'='*70}")
            print(f"BATCH {batch_num}/{total_batches} (images {i+1} to {min(i+batch_size, total_urls)})")
            print('='*70)

            download_batch(batch, output_dir, max_workers=20, timeout=10)

        overall_elapsed = time.time() - overall_start

        # Count actual files
        downloaded_count = (
            len(list(output_dir.glob("*.jpg"))) +
            len(list(output_dir.glob("*.jpeg"))) +
            len(list(output_dir.glob("*.png"))) +
            len(list(output_dir.glob("*.JPG"))) +
            len(list(output_dir.glob("*.JPEG"))) +
            len(list(output_dir.glob("*.PNG")))
        )

        print()
        print("=" * 70)
        print("DOWNLOAD SUMMARY")
        print("=" * 70)
        print(f"  URLs found: {total_urls}")
        print(f"  Successfully downloaded: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Timeouts: {stats['timeouts']}")
        print(f"  Skipped (already existed): {stats['skipped']}")
        print(f"  Files on disk: {downloaded_count}")
        print(f"  Total time: {overall_elapsed:.1f}s ({total_urls/overall_elapsed:.1f} images/s)")
        print()

        # Write log
        with open(log_file, 'w') as f:
            f.write(f"GBIF Download Log - {species_name}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Taxon Key: {taxon_key}\n")
            f.write(f"Download time: {datetime.now()}\n")
            f.write(f"Total URLs found: {total_urls}\n")
            f.write(f"Successfully downloaded: {stats['success']}\n")
            f.write(f"Failed downloads: {stats['failed']}\n")
            f.write(f"Timeouts: {stats['timeouts']}\n")
            f.write(f"Skipped (already existed): {stats['skipped']}\n")
            f.write(f"Files on disk: {downloaded_count}\n")
            f.write(f"Total time: {overall_elapsed:.1f}s\n")
            if total_urls > 0:
                f.write(f"Success rate: {(downloaded_count / total_urls * 100):.2f}%\n")

        return downloaded_count

    except Exception as e:
        print(f"ERROR downloading {species_name}: {e}")
        import traceback
        traceback.print_exc()
        with open(log_file, 'w') as f:
            f.write(f"ERROR: {str(e)}\n")
            f.write(f"Time: {datetime.now()}\n")
        return 0


def main():
    """Main download function"""

    print("\n" + "=" * 70)
    print("GBIF IMAGE COLLECTION V2 - SETUP")
    print("=" * 70)
    print("\nTarget Species:")
    print("  1. European Hornet (Vespa crabro) - taxon_key: 1311527")
    print("  2. Wasps (Vespidae family) - taxon_key: 52747")
    print("\nFeatures:")
    print("  - 10 second timeout per image")
    print("  - 20 concurrent downloads")
    print("  - Resume capability (skips existing files)")
    print("  - Better error handling")
    print("\nTarget: Up to 50,000 images per species")
    print()

    # Setup directories
    european_dir, wasps_dir = setup_directories()

    # Track results
    results = {}

    # Download European Hornets
    print("\n" + "=" * 70)
    print("PHASE 1: EUROPEAN HORNETS")
    print("=" * 70)
    print()

    european_count = download_species(
        taxon_key=1311527,  # Vespa crabro
        species_name="European Hornet (Vespa crabro)",
        output_dir=european_dir,
        max_images=50000,
        batch_size=1000
    )
    results['european_hornets'] = european_count

    # Download Wasps
    print("\n" + "=" * 70)
    print("PHASE 2: WASPS")
    print("=" * 70)
    print()

    wasps_count = download_species(
        taxon_key=52747,  # Vespidae family
        species_name="Wasps (Vespidae family)",
        output_dir=wasps_dir,
        max_images=50000,
        batch_size=1000
    )
    results['wasps'] = wasps_count

    # Final summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE - FINAL SUMMARY")
    print("=" * 70)
    print(f"\nEuropean Hornets: {results['european_hornets']:,} images")
    print(f"Wasps: {results['wasps']:,} images")
    print(f"Total: {sum(results.values()):,} images")
    print()
    print("Images saved to:")
    print(f"  {european_dir}")
    print(f"  {wasps_dir}")
    print()

    # Save summary JSON
    summary_file = Path("D:/Ultimate Dataset/gbif_download_summary_v2.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'download_time': datetime.now().isoformat(),
            'results': results,
            'total_images': sum(results.values()),
            'directories': {
                'european_hornets': str(european_dir),
                'wasps': str(wasps_dir)
            }
        }, f, indent=2)

    print(f"Summary saved to: {summary_file}")
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Run validate_downloaded_images.py to check image validity")
    print("2. Remove duplicates and corrupted images")
    print("3. Integrate with existing training dataset")
    print("=" * 70)


if __name__ == "__main__":
    main()
