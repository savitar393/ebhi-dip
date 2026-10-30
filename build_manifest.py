import csv
from pathlib import Path

project_root = Path(__file__).resolve().parent
dataset_root = project_root / "data" / "raw" / "EBHI-SEG"

rows = []

for category in sorted(dataset_root.iterdir()):
    if not category.is_dir():
        continue

    for image_path in sorted((category / "image").glob("*.png")):
        mask_path = category / "label" / image_path.name

        rows.append({
            "sample_id": f"{category.name}/{image_path.stem}",
            "category": category.name,
            "image_path": image_path.relative_to(project_root).as_posix(),
            "mask_path": mask_path.relative_to(project_root).as_posix(),
            "pair_status": (
                "paired" if mask_path.is_file() else "missing_mask"
            ),
        })

output_path = project_root / "data" / "metadata" / "manifest.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

columns = [
    "sample_id",
    "category",
    "image_path",
    "mask_path",
    "pair_status",
]

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

paired = sum(row["pair_status"] == "paired" for row in rows)

print("Total image records:", len(rows))
print("Paired:", paired)
print("Missing masks:", len(rows) - paired)
print("Saved:", output_path)