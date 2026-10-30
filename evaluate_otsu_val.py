import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"

config_path = project_root / "configs" / "otsu_raw.json"
with config_path.open(encoding="utf-8") as file:
    config = json.load(file)

supported_settings = {
    "method": "otsu",
    "grayscale_conversion": "RGB2GRAY",
    "threshold_mode": "per_image_otsu",
    "postprocessing": "none",
    "polarity_selected_on": "train",
}

for key, expected in supported_settings.items():
    if config.get(key) != expected:
        raise SystemExit(f"Unsupported configuration: {key}")

polarity = config["foreground_polarity"]
if polarity not in ("dark", "bright"):
    raise SystemExit("Unknown foreground polarity.")

split_path = metadata_dir / "split_manifest.csv"
actual_checksum = hashlib.sha256(split_path.read_bytes()).hexdigest()
saved_checksum = (
    (metadata_dir / "split_manifest.sha256")
    .read_text(encoding="utf-8")
    .split()[0]
)

if (
    actual_checksum != saved_checksum
    or actual_checksum != config["split_sha256"]
):
    raise SystemExit("Split checksum does not match.")

with split_path.open(newline="", encoding="utf-8-sig") as file:
    records = [
        row for row in csv.DictReader(file)
        if row["split"] == "val"
    ]


def overlap_scores(predicted, reference):
    intersection = np.count_nonzero(predicted & reference)
    union = np.count_nonzero(predicted | reference)
    total = (
        np.count_nonzero(predicted)
        + np.count_nonzero(reference)
    )

    dice = 2 * intersection / total if total else 1.0
    iou = intersection / union if union else 1.0
    return dice, iou


results = []

for record in records:
    with Image.open(project_root / record["image_path"]) as source:
        image = np.array(source.convert("RGB"))

    with Image.open(project_root / record["mask_path"]) as source:
        reference = np.array(source) >= config["reference_threshold"]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold, bright_u8 = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    bright = bright_u8 > 0
    predicted = ~bright if polarity == "dark" else bright
    all_foreground = np.ones_like(gray, dtype=bool)

    dice, iou = overlap_scores(predicted, reference)
    control_dice, control_iou = overlap_scores(all_foreground, reference)

    results.append({
        "sample_id": record["sample_id"],
        "category": record["category"],
        "group_id": record["group_id"],
        "foreground_polarity": polarity,
        "otsu_threshold": threshold,
        "otsu_dice": dice,
        "otsu_iou": iou,
        "all_foreground_dice": control_dice,
        "all_foreground_iou": control_iou,
    })

output_dir = project_root / "results"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "otsu_val.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(results)

print("OpenCV:", cv2.__version__)
print("Fixed foreground polarity:", polarity)
print("Validation images evaluated:", len(results))
print("\nMean Dice within each category:")
print(f"{'Category':<20} {'Images':>6} {'Otsu':>9} {'All FG':>9}")

category_means = {}

for category in sorted({row["category"] for row in results}):
    subset = [
        row for row in results
        if row["category"] == category
    ]

    for method in ("otsu", "all_foreground"):
        category_means[(category, method)] = (
            float(np.mean([row[f"{method}_dice"] for row in subset])),
            float(np.mean([row[f"{method}_iou"] for row in subset])),
        )

    print(
        f"{category:<20} {len(subset):>6} "
        f"{category_means[(category, 'otsu')][0]:>9.4f} "
        f"{category_means[(category, 'all_foreground')][0]:>9.4f}"
    )

categories = sorted({row["category"] for row in results})

print("\nEqual-weight mean across categories:")
for method in ("otsu", "all_foreground"):
    dice = np.mean([
        category_means[(category, method)][0]
        for category in categories
    ])
    iou = np.mean([
        category_means[(category, method)][1]
        for category in categories
    ])
    print(f"  {method}: Dice={dice:.4f}, IoU={iou:.4f}")

print("\nSaved:", output_path)