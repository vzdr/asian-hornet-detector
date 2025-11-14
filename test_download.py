"""
Quick test - download just 100 images to verify it works
"""

from pathlib import Path
from gbif_dl import api
import requests
import hashlib

output_dir = Path("D:/Ultimate Dataset/european_hornets_gbif")

print("Testing download of 100 European Hornet images...")
print(f"Output: {output_dir}\n")

# Get just 100 URLs
media_generator = api.gbif_query_generator(
    taxon_key="1311527",
    media_type='StillImage'
)

media_list = []
for i, media_data in enumerate(media_generator):
    if i >= 100:
        break
    media_list.append(media_data)
    if (i+1) % 20 == 0:
        print(f"Collected {i+1} URLs...")

print(f"\nCollected {len(media_list)} URLs")
print("Starting downloads...\n")

success = 0
failed = 0
skipped = 0

for i, media_data in enumerate(media_list):
    try:
        url = media_data['url']
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = url.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png']:
            ext = 'jpg'

        gbif_id = media_data['label'].get('gbifID', url_hash)
        filename = f"{gbif_id}_{url_hash}.{ext}"
        filepath = output_dir / filename

        if filepath.exists() and filepath.stat().st_size > 0:
            skipped += 1
            continue

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            f.write(response.content)

        if filepath.stat().st_size > 0:
            success += 1
            if success % 10 == 0:
                print(f"Downloaded {success} images...")
        else:
            filepath.unlink()
            failed += 1

    except Exception as e:
        failed += 1
        print(f"Failed: {str(e)[:50]}")

print(f"\nResults:")
print(f"  Success: {success}")
print(f"  Failed: {failed}")
print(f"  Skipped: {skipped}")
print(f"  Total new files: {success}")
