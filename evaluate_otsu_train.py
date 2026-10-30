import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import json

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"
split_path = metadata_dir / "split_manifest.csv"

expected_checksum = (
    (metadata_dir / "split_manifest.sha256")
    .read_text(encoding="utf-8")
    .split()[0]
)

if hashlib.sha256(split_path.read_bytes()).hexdigest() != expected_checksum:
    raise SystemExit("Frozen split checksum does not match.")

with split_path.open(newline="", encoding="utf-8-sig") as file:
    records = [
        row for row in csv.DictReader(file)
        if row["split"] == "train"
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


methods = ("dark", "bright", "all_foreground")
results = []

for record in records:
    with Image.open(project_root / record["image_path"]) as source:
        image = np.array(source.convert("RGB"))

    with Image.open(project_root / record["mask_path"]) as source:
        reference = np.array(source) >= 128

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    threshold, bright_u8 = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    bright = bright_u8 > 0

    predictions = [
        ("dark", ~bright),
        ("bright", bright),
        ("all_foreground", np.ones_like(gray, dtype=bool)),
    ]

    result = {
        "sample_id": record["sample_id"],
        "category": record["category"],
        "group_id": record["group_id"],
        "otsu_threshold": threshold,
        "reference_threshold": 128,
    }

    for name, predicted in predictions:
        dice, iou = overlap_scores(predicted, reference)
        result[f"{name}_dice"] = dice
        result[f"{name}_iou"] = iou

    results.append(result)

output_dir = project_root / "results"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "otsu_train.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(results)

print("OpenCV:", cv2.__version__)
print("Training images evaluated:", len(results))
print("\nMean Dice within each category:")
print(
    f"{'Category':<20} {'Images':>6} "
    f"{'Dark':>9} {'Bright':>9} {'All FG':>9}"
)

category_means = {}

for category in sorted({row["category"] for row in results}):
    subset = [
        row for row in results
        if row["category"] == category
    ]

    for name in methods:
        category_means[(category, name)] = (
            float(np.mean([row[f"{name}_dice"] for row in subset])),
            float(np.mean([row[f"{name}_iou"] for row in subset])),
        )

    print(
        f"{category:<20} {len(subset):>6} "
        f"{category_means[(category, 'dark')][0]:>9.4f} "
        f"{category_means[(category, 'bright')][0]:>9.4f} "
        f"{category_means[(category, 'all_foreground')][0]:>9.4f}"
    )

categories = sorted({row["category"] for row in results})

print("\nEqual-weight mean across categories:")
for name in methods:
    dice = np.mean([
        category_means[(category, name)][0]
        for category in categories
    ])
    iou = np.mean([
        category_means[(category, name)][1]
        for category in categories
    ])

    print(f"  {name}: Dice={dice:.4f}, IoU={iou:.4f}")

print("\nSaved:", output_path)

polarity_scores = {}

for name in ("dark", "bright"):
    polarity_scores[name] = float(np.mean([
        category_means[(category, name)][0]
        for category in categories
    ]))

chosen_polarity = max(polarity_scores, key=polarity_scores.get)

config = {
    "method": "otsu",
    "grayscale_conversion": "RGB2GRAY",
    "threshold_mode": "per_image_otsu",
    "foreground_polarity": chosen_polarity,
    "postprocessing": "none",
    "reference_threshold": 128,
    "polarity_selected_on": "train",
    "selection_metric": "mean_category_dice",
    "training_mean_category_dice": polarity_scores[chosen_polarity],
    "split_sha256": expected_checksum,
    "opencv_version": cv2.__version__,
    "numpy_version": np.__version__,
}

config_dir = project_root / "configs"
config_dir.mkdir(parents=True, exist_ok=True)
config_path = config_dir / "otsu_raw.json"

with config_path.open("w", encoding="utf-8", newline="\n") as file:
    json.dump(config, file, indent=2)
    file.write("\n")

print("\nChosen Otsu polarity:", chosen_polarity)
print("Saved configuration:", config_path)