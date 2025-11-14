"""
Debug script to inspect GBIF MediaData structure
"""

from gbif_dl import api

print("Querying GBIF for European Hornet images...")

media_generator = api.gbif_query_generator(
    taxon_key="1311527",
    media_type='StillImage'
)

print("\nInspecting first 5 MediaData objects:\n")

for i, media_data in enumerate(media_generator):
    if i >= 5:
        break

    print(f"\n{'='*70}")
    print(f"MediaData Object {i+1}:")
    print(f"{'='*70}")
    print(f"Type: {type(media_data)}")
    print(f"\nDictionary Keys: {list(media_data.keys())}")
    print(f"\nFull Dictionary:")
    for key, value in media_data.items():
        print(f"  {key}: {value}")

print("\n\nDone inspecting MediaData objects.")
