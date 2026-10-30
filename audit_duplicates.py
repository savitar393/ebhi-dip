import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"

with (metadata_dir / "manifest.csv").open(
    newline="", encoding="utf-8-sig"
) as file:
    records = list(csv.DictReader(file))

groups = defaultdict(list)
results = []

for record in records:
    image_path = project_root / record["image_path"]

    with Image.open(image_path) as source:
        image = source.convert("RGB")

        fingerprint = hashlib.sha256()
        fingerprint.update(
            f"RGB:{image.width}:{image.height}:".encode("ascii")
        )
        fingerprint.update(image.tobytes())

    pixel_hash = fingerprint.hexdigest()

    groups[pixel_hash].append(record)
    results.append({
        **record,
        "pixel_sha256": pixel_hash,
    })

duplicate_groups = [
    members for members in groups.values()
    if len(members) > 1
]

cross_category_groups = [
    members for members in duplicate_groups
    if len({row["category"] for row in members}) > 1
]

output_path = metadata_dir / "image_hashes.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

print("Images scanned:", len(results))
print("Distinct pixel fingerprints:", len(groups))
print("Duplicate groups:", len(duplicate_groups))
print(
    "Images belonging to duplicate groups:",
    sum(len(members) for members in duplicate_groups),
)
print(
    "Extra copies beyond one per group:",
    sum(len(members) - 1 for members in duplicate_groups),
)
print("Cross-category duplicate groups:", len(cross_category_groups))
print("Saved:", output_path)

print("\nExample duplicate groups:")
for number, members in enumerate(duplicate_groups[:5], start=1):
    print(f"\nGroup {number}:")
    for row in members:
        print(f"  {row['sample_id']} ({row['pair_status']})")