import csv
from pathlib import Path

import numpy as np
from PIL import Image

from collections import Counter

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"

with (metadata_dir / "manifest.csv").open(
    newline="", encoding="utf-8-sig"
) as file:
    records = list(csv.DictReader(file))

results = []

for record in records:
    if record["pair_status"] != "paired":
        continue

    with Image.open(project_root / record["mask_path"]) as source:
        mask = np.array(source)

    gray_pixels = (mask > 0) & (mask < 255)
    binary_mask = mask >= 128

    results.append({
        "sample_id": record["sample_id"],
        "category": record["category"],
        "minimum": int(mask.min()),
        "maximum": int(mask.max()),
        "unique_values": int(np.unique(mask).size),
        "gray_pixels": int(np.count_nonzero(gray_pixels)),
        "gray_percentage": float(100 * gray_pixels.mean()),
        "foreground_pixels": int(np.count_nonzero(binary_mask)),
        "foreground_percentage": float(100 * binary_mask.mean()),
    })

if not results:
    raise SystemExit("No paired masks found in the manifest.")

output_path = metadata_dir / "mask_audit.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

gray_percentages = [row["gray_percentage"] for row in results]
foreground_percentages = [
    row["foreground_percentage"] for row in results
]

print("Masks audited:", len(results))
print(
    "Masks containing only 0 and/or 255:",
    sum(row["gray_pixels"] == 0 for row in results),
)
print(
    "Masks containing intermediate gray values:",
    sum(row["gray_pixels"] > 0 for row in results),
)
print(
    "Intermediate gray percentage (min / mean / max): "
    f"{min(gray_percentages):.2f}% / "
    f"{np.mean(gray_percentages):.2f}% / "
    f"{max(gray_percentages):.2f}%"
)
print(
    "Foreground coverage at threshold 128 (min / max): "
    f"{min(foreground_percentages):.2f}% / "
    f"{max(foreground_percentages):.2f}%"
)
print(
    "Entirely background after threshold:",
    sum(row["foreground_pixels"] == 0 for row in results),
)
print(
    "Entirely foreground after threshold:",
    sum(row["foreground_pixels"] == 224 * 224 for row in results),
)
print("Saved:", output_path)

full_masks = [
    row for row in results
    if row["foreground_pixels"] == 224 * 224
]

constant_masks = [
    row for row in full_masks
    if row["minimum"] == row["maximum"]
]

print("\nFull-foreground masks by category:")
category_counts = Counter(row["category"] for row in full_masks)

for category, count in sorted(category_counts.items()):
    print(f"  {category}: {count}")

print("\nConstant full-foreground masks:", len(constant_masks))
print(
    "Nonconstant full-foreground masks:",
    len(full_masks) - len(constant_masks),
)

print("\nPixel values of constant full-foreground masks:")
value_counts = Counter(row["minimum"] for row in constant_masks)

for value, count in sorted(value_counts.items()):
    print(f"  value={value}: {count} masks")