"""
Download European Hornet and Wasp images from GBIF
Uses gbif-dl library to download images with open licenses
Saves to D:/Ultimate Dataset/ in separate folders
"""

import os
from pathlib import Path
from datetime import datetime
import json

# Import gbif-dl components correctly
from gbif_dl import api, dl_async

def setup_directories():
    """Create directory structure for downloads"""
    base_dir = Path("D:/Ultimate Dataset")

    european_hornets_dir = base_dir / "european_hornets_gbif"
    wasps_dir = base_dir / "wasps_gbif"

    european_hornets_dir.mkdir(parents=True, exist_ok=True)
    wasps_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GBIF IMAGE DOWNLOAD - EUROPEAN HORNETS & WASPS")
    print("=" * 70)
    print(f"\nDirectories created:")
    print(f"  European Hornets: {european_hornets_dir}")
    print(f"  Wasps: {wasps_dir}")
    print()

    return european_hornets_dir, wasps_dir


def download_species(taxon_key, species_name, output_dir, max_images=50000):
    """
    Download images for a specific species from GBIF

    Args:
        taxon_key: GBIF taxon key
        species_name: Name for logging
        output_dir: Where to save images
        max_images: Maximum number of images to download
    """
    print("=" * 70)
    print(f"DOWNLOADING: {species_name}")
    print("=" * 70)
    print(f"Taxon Key: {taxon_key}")
    print(f"Output Directory: {output_dir}")
    print(f"Max Images: {max_images}")
    print()

    log_file = output_dir / "download_log.txt"

    try:
        # Generate MediaData objects from GBIF API
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Querying GBIF API for image URLs...")

        # Create query generator for occurrences with images
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
            if len(media_data_list) % 100 == 0:
                print(f"  Collected {len(media_data_list)} media items...")

        total_urls = len(media_data_list)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {total_urls} image URLs")

        if total_urls == 0:
            print(f"WARNING: No images found for {species_name}")
            with open(log_file, 'w') as f:
                f.write(f"No images found for {species_name} (taxon_key={taxon_key})\n")
                f.write(f"Query time: {datetime.now()}\n")
            return 0

        # Download images using dl_async
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting download...")
        print(f"  Using async multi-threaded download")
        print()

        # Download with gbif-dl's async downloader
        dl_async.download(
            items=media_data_list,
            root=str(output_dir),
            nb_workers=10  # Parallel downloads
        )

        # Count successful downloads
        downloaded_count = (
            len(list(output_dir.glob("*.jpg"))) +
            len(list(output_dir.glob("*.jpeg"))) +
            len(list(output_dir.glob("*.png"))) +
            len(list(output_dir.glob("*.JPG"))) +
            len(list(output_dir.glob("*.JPEG"))) +
            len(list(output_dir.glob("*.PNG")))
        )

        print()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Download complete!")
        print(f"  URLs found: {total_urls}")
        print(f"  Images downloaded: {downloaded_count}")
        print(f"  Failed: {total_urls - downloaded_count}")

        # Write log
        with open(log_file, 'w') as f:
            f.write(f"GBIF Download Log - {species_name}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Taxon Key: {taxon_key}\n")
            f.write(f"Query time: {datetime.now()}\n")
            f.write(f"Total URLs found: {total_urls}\n")
            f.write(f"Images successfully downloaded: {downloaded_count}\n")
            f.write(f"Failed downloads: {total_urls - downloaded_count}\n")
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
    print("GBIF IMAGE COLLECTION - SETUP")
    print("=" * 70)
    print("\nTarget Species:")
    print("  1. European Hornet (Vespa crabro) - taxon_key: 1311527")
    print("  2. Wasps (Vespidae family) - taxon_key: 52747")
    print("\nLicenses: CC0, CC-BY, CC-BY-NC (open licenses)")
    print("Target: Up to 50,000 images per species (or as many as available)")
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
        max_images=50000
    )
    results['european_hornets'] = european_count

    print("\n" + "=" * 70)
    print("PHASE 2: WASPS")
    print("=" * 70)
    print()

    # Download Wasps (as many as available)
    wasps_count = download_species(
        taxon_key=52747,  # Vespidae family (corrected taxon_key)
        species_name="Wasps (Vespidae family)",
        output_dir=wasps_dir,
        max_images=50000
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
    summary_file = Path("D:/Ultimate Dataset/gbif_download_summary.json")
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
