"""
GBIF Chunked Downloader - Process and download in chunks
Collects 1000 URLs at a time, downloads them immediately, then continues
"""

from pathlib import Path
from gbif_dl import api
import requests
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

def download_single_image(media_data, output_dir):
    """Download a single image"""
    try:
        url = media_data['url']
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = url.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
            ext = 'jpg'

        gbif_id = media_data['label'].get('gbifID', url_hash)
        filename = f"{gbif_id}_{url_hash}.{ext}"
        filepath = output_dir / filename

        # Skip if exists
        if filepath.exists() and filepath.stat().st_size > 0:
            return "skipped"

        # Download with timeout
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if filepath.stat().st_size > 0:
            return "success"
        else:
            filepath.unlink()
            return "empty"

    except requests.exceptions.Timeout:
        return "timeout"
    except Exception as e:
        return f"error"

def download_chunk(media_list, output_dir, workers=20):
    """Download a chunk of images"""
    stats = {"success": 0, "failed": 0, "skipped": 0, "timeout": 0}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_single_image, m, output_dir): m for m in media_list}

        for future in as_completed(futures):
            result = future.result()
            if result == "success":
                stats["success"] += 1
            elif result == "skipped":
                stats["skipped"] += 1
            elif result == "timeout":
                stats["timeout"] += 1
                stats["failed"] += 1
            else:
                stats["failed"] += 1

    return stats

def main():
    output_dir = Path("D:/Ultimate Dataset/european_hornets_gbif")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GBIF CHUNKED DOWNLOADER")
    print("="*70)
    print(f"\nOutput: {output_dir}")
    print(f"Target: 50,000 European Hornet images")
    print(f"Strategy: Collect & download in 1000-image chunks")
    print()

    media_generator = api.gbif_query_generator(
        taxon_key="1311527",
        media_type='StillImage'
    )

    total_stats = {"success": 0, "failed": 0, "skipped": 0, "timeout": 0}
    chunk_size = 1000
    max_images = 50000
    collected = 0

    chunk = []
    chunk_num = 1

    print(f"Starting collection and download...")
    print()

    for media_data in media_generator:
        chunk.append(media_data)
        collected += 1

        if len(chunk) >= chunk_size or collected >= max_images:
            print(f"{'='*70}")
            print(f"CHUNK {chunk_num} - Collected {collected}/{max_images} URLs")
            print(f"{'='*70}")
            print(f"Downloading {len(chunk)} images...")

            start = datetime.now()
            stats = download_chunk(chunk, output_dir, workers=20)
            elapsed = (datetime.now() - start).total_seconds()

            # Update totals
            for key in total_stats:
                total_stats[key] += stats[key]

            print(f"  Success: {stats['success']}")
            print(f"  Failed: {stats['failed']} (Timeouts: {stats['timeout']})")
            print(f"  Skipped: {stats['skipped']}")
            print(f"  Time: {elapsed:.1f}s ({len(chunk)/elapsed:.1f} imgs/s)")
            print(f"  Total downloaded so far: {total_stats['success']}")
            print()

            chunk = []
            chunk_num += 1

        if collected >= max_images:
            break

    # Download any remaining
    if chunk:
        print(f"{'='*70}")
        print(f"FINAL CHUNK - Downloading {len(chunk)} images...")
        print(f"{'='*70}")
        stats = download_chunk(chunk, output_dir, workers=20)
        for key in total_stats:
            total_stats[key] += stats[key]

    print()
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"\nTotal URLs processed: {collected}")
    print(f"Successfully downloaded: {total_stats['success']}")
    print(f"Failed: {total_stats['failed']} (Timeouts: {total_stats['timeout']})")
    print(f"Skipped (already existed): {total_stats['skipped']}")
    print()

    # Count actual files
    file_count = len(list(output_dir.glob("*.jpg"))) + len(list(output_dir.glob("*.jpeg"))) + len(list(output_dir.glob("*.png")))
    print(f"Total files on disk: {file_count}")

    # Save log
    log_file = output_dir / "download_log_chunked.txt"
    with open(log_file, 'w') as f:
        f.write(f"GBIF Chunked Download Log\n")
        f.write(f"={'='*50}\n\n")
        f.write(f"Download time: {datetime.now()}\n")
        f.write(f"URLs processed: {collected}\n")
        f.write(f"Successfully downloaded: {total_stats['success']}\n")
        f.write(f"Failed: {total_stats['failed']}\n")
        f.write(f"Timeouts: {total_stats['timeout']}\n")
        f.write(f"Skipped: {total_stats['skipped']}\n")
        f.write(f"Files on disk: {file_count}\n")

    print(f"\nLog saved to: {log_file}")
    print("="*70)

if __name__ == "__main__":
    main()
