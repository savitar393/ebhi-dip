import csv
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"

with (metadata_dir / "manifest.csv").open(
    newline="", encoding="utf-8-sig"
) as file:
    records = [
        row for row in csv.DictReader(file)
        if row["pair_status"] == "paired"
    ]

groups = defaultdict(list)

for record in records:
    filename = Path(record["image_path"]).stem
    prefix = filename.split("-")[0]
    groups[prefix].append(record)

summary = []

for prefix, members in sorted(groups.items()):
    categories = sorted({row["category"] for row in members})

    summary.append({
        "filename_prefix": prefix,
        "image_count": len(members),
        "category_count": len(categories),
        "categories": " | ".join(categories),
    })

output_path = metadata_dir / "filename_groups.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print("Paired images:", len(records))
print("Distinct filename prefixes:", len(groups))
print(
    "Prefixes containing multiple images:",
    sum(row["image_count"] > 1 for row in summary),
)
print(
    "Prefixes appearing across categories:",
    sum(row["category_count"] > 1 for row in summary),
)

print("\nImages and distinct prefixes per category:")
for category in sorted({row["category"] for row in records}):
    category_records = [
        row for row in records if row["category"] == category
    ]
    prefixes = {
        Path(row["image_path"]).stem.split("-")[0]
        for row in category_records
    }
    print(
        f"  {category}: images={len(category_records)}, "
        f"prefixes={len(prefixes)}"
    )

print("\nFive largest filename groups:")
for row in sorted(
    summary, key=lambda row: (-row["image_count"], row["filename_prefix"])
)[:5]:
    print(
        f"  {row['filename_prefix']}: "
        f"{row['image_count']} images; {row['categories']}"
    )

print("\nSaved:", output_path)

case_variants = defaultdict(list)

for prefix in groups:
    case_variants[prefix.upper()].append(prefix)

case_collisions = [
    sorted(variants)
    for variants in case_variants.values()
    if len(variants) > 1
]

print("\nPrefixes when compared without letter case:", len(case_variants))
print("Prefixes with multiple capitalization variants:", len(case_collisions))

for variants in sorted(case_collisions)[:10]:
    members = [
        record
        for prefix in variants
        for record in groups[prefix]
    ]
    categories = sorted({row["category"] for row in members})

    print("\nVariants:", " | ".join(variants))
    print("  Images:", len(members))
    print("  Categories:", " | ".join(categories))