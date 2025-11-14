"""
Check what wasp-related taxon keys actually have images
"""

from gbif_dl import api

print("Testing different wasp taxon keys:\n")

# Try different wasp-related taxon keys
test_keys = {
    "52747": "Vespidae family (attempt 1)",
    "4490": "Vespidae family (GBIF backbone)",
    "1311334": "Vespa genus",
    "1311396": "Vespula genus (common wasps)",
    "5204144": "Polistes genus (paper wasps)",
}

for key, name in test_keys.items():
    print(f"\nTesting {key} - {name}:")
    try:
        media_gen = api.gbif_query_generator(
            taxon_key=key,
            media_type='StillImage'
        )

        count = 0
        for media in media_gen:
            count += 1
            if count >= 10:
                break

        print(f"  SUCCESS - Found images! (tested first 10)")

        if count > 0:
            print(f"  Sample URL: {media['url']}")
            break
    except Exception as e:
        print(f"  FAILED - Error: {e}")

print("\nDone testing")
