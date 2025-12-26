"""
Quick script to find sample images from each class for inference testing
"""
from pathlib import Path
import random

# Data sources
data_sources = {
    'bees_hornets1': r'D:\Ultimate Dataset\BeesAndHornets1\Bee And Asian Hornet Detection',
    'bees_hornets2': r'D:\Ultimate Dataset\BeesAndHornets2\Dataset',
    'gbif_european_hornets': r'D:\Ultimate Dataset\european_hornets_gbif',
    'gbif_wasps': r'D:\Ultimate Dataset\wasps_gbif'
}

# Find images for each class
samples = {
    'asian_hornet': [],
    'bee': [],
    'european_hornet': [],
    'wasp': []
}

# Asian hornets from BeesAndHornets1
asian_path = Path(data_sources['bees_hornets1']) / 'valid' / 'asian_hornet'
if asian_path.exists():
    asian_images = list(asian_path.glob('*.jpg')) + list(asian_path.glob('*.jpeg'))
    samples['asian_hornet'] = random.sample(asian_images, min(2, len(asian_images)))

# Bees from BeesAndHornets1
bee_path = Path(data_sources['bees_hornets1']) / 'valid' / 'bee'
if bee_path.exists():
    bee_images = list(bee_path.glob('*.jpg')) + list(bee_path.glob('*.jpeg'))
    samples['bee'] = random.sample(bee_images, min(2, len(bee_images)))

# European hornets from GBIF
european_path = Path(data_sources['gbif_european_hornets'])
if european_path.exists():
    european_images = list(european_path.glob('*.jpg')) + list(european_path.glob('*.jpeg'))
    samples['european_hornet'] = random.sample(european_images, min(2, len(european_images)))

# Wasps from GBIF
wasp_path = Path(data_sources['gbif_wasps'])
if wasp_path.exists():
    wasp_images = list(wasp_path.glob('*.jpg')) + list(wasp_path.glob('*.jpeg'))
    samples['wasp'] = random.sample(wasp_images, min(2, len(wasp_images)))

# Print results
print("Sample images for inference testing:")
print("=" * 70)
for class_name, images in samples.items():
    print(f"\n{class_name.replace('_', ' ').title()}:")
    for img in images:
        print(f"  {img}")

# Save to file
output_file = Path('sample_images.txt')
with open(output_file, 'w') as f:
    for class_name, images in samples.items():
        f.write(f"{class_name}:\n")
        for img in images:
            f.write(f"  {img}\n")

print(f"\n\nSample paths saved to: {output_file}")
