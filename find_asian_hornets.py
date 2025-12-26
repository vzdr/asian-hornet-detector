from pathlib import Path

label_dir = Path("D:/Ultimate Dataset/BeesAndHornets2/Dataset/train/labels")
img_dir = Path("D:/Ultimate Dataset/BeesAndHornets2/Dataset/train/images")

count = 0
asian_hornets = []

for label_file in label_dir.glob("*.txt"):
    with open(label_file) as f:
        first_line = f.readline().strip()
        if first_line:
            class_id = int(first_line.split()[0])
            if class_id == 1:
                img_name = label_file.stem + ".jpg"
                img_path = img_dir / img_name
                if img_path.exists():
                    asian_hornets.append(str(img_path))
                    count += 1
                    if count >= 10:
                        break

print(f"Found {len(asian_hornets)} Asian hornet images:")
for img in asian_hornets:
    print(f"  {img}")
