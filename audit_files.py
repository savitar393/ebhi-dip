import csv
from collections import Counter
from pathlib import Path

from PIL import Image

project_root = Path(__file__).resolve().parent
manifest_path = project_root / "data" / "metadata" / "manifest.csv"


def read_metadata(relative_path):
    path = project_root / relative_path

    with Image.open(path) as source:
        source.load()
        return source.size, source.mode


with manifest_path.open(newline="", encoding="utf-8-sig") as file:
    records = list(csv.DictReader(file))

image_formats = Counter()
mask_formats = Counter()
read_errors = []
size_mismatches = []
skipped = 0
decoded = 0

for record in records:
    if record["pair_status"] != "paired":
        skipped += 1
        continue

    try:
        image_size, image_mode = read_metadata(record["image_path"])
        mask_size, mask_mode = read_metadata(record["mask_path"])
    except (OSError, ValueError) as error:
        read_errors.append((record["sample_id"], str(error)))
        continue

    decoded += 1
    image_formats[(image_size, image_mode)] += 1
    mask_formats[(mask_size, mask_mode)] += 1

    if image_size != mask_size:
        size_mismatches.append(
            (record["sample_id"], image_size, mask_size)
        )

print("Manifest records:", len(records))
print("Skipped unpaired records:", skipped)
print("Successfully decoded pairs:", decoded)
print("Pairs with read errors:", len(read_errors))
print("Pairs with different dimensions:", len(size_mismatches))

print("\nImage formats:")
for (size, mode), count in sorted(image_formats.items()):
    print(f"  size={size}, mode={mode}: {count}")

print("\nMask formats:")
for (size, mode), count in sorted(mask_formats.items()):
    print(f"  size={size}, mode={mode}: {count}")

for sample_id, error in read_errors[:10]:
    print(f"\nRead error: {sample_id}: {error}")

for sample_id, image_size, mask_size in size_mismatches[:10]:
    print(f"\nSize mismatch: {sample_id}: {image_size} vs {mask_size}")