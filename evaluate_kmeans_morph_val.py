import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

from kmeans_prediction import nearest_clusters
from postprocessing import morphology_variants


REFERENCE_THRESHOLD = 128
STAGES = ("M0", "M1", "M2")

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"
split_path = metadata_dir / "split_manifest.csv"

expected_checksum = (
    (metadata_dir / "split_manifest.sha256")
    .read_text(encoding="utf-8")
    .split()[0]
)

if hashlib.sha256(split_path.read_bytes()).hexdigest() != expected_checksum:
    raise SystemExit("Frozen split checksum mismatch.")

with split_path.open(newline="", encoding="utf-8-sig") as file:
    validation_records = [
        row for row in csv.DictReader(file)
        if row["split"] == "val"
    ]

if len(validation_records) != 372:
    raise SystemExit("Expected 372 validation images.")

category_counts = Counter(
    row["category"] for row in validation_records
)
categories = sorted(category_counts)

helper_checksum = hashlib.sha256(
    (project_root / "kmeans_prediction.py").read_bytes()
).hexdigest()

postprocessing_checksum = hashlib.sha256(
    (project_root / "postprocessing.py").read_bytes()
).hexdigest()

models = {}
model_checksums = {}

for k in (2, 3, 4):
    model_path = (
        project_root / "models" / f"kmeans_lab_k{k}_mapped.json"
    )
    model_bytes = model_path.read_bytes()
    model = json.loads(model_bytes)

    if (
        model["fit_subset"] != "train"
        or model["split_sha256"] != expected_checksum
        or model["prediction"]["helper_sha256"] != helper_checksum
        or model["foreground_mapping"]["reference_threshold"]
        != REFERENCE_THRESHOLD
        or model["foreground_mapping"]["threshold"] != 0.5
        or model["sampling"]["opencv_version"] != cv2.__version__
    ):
        raise SystemExit(f"Configuration mismatch for k={k}.")

    models[k] = model
    model_checksums[f"k{k}"] = hashlib.sha256(
        model_bytes
    ).hexdigest()


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

for number, record in enumerate(validation_records, start=1):
    with Image.open(project_root / record["image_path"]) as source:
        image = np.array(source.convert("RGB"))

    rgb_float = image.astype(np.float32) / np.float32(255.0)
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)

    masks = {}

    for k, model in models.items():
        labels = nearest_clusters(
            lab.reshape(-1, 3),
            model["cluster_centers"],
        ).reshape(image.shape[:2])

        raw_mask = np.isin(
            labels, model["foreground_clusters"]
        )

        variants = morphology_variants(raw_mask)

        for stage in STAGES:
            masks[f"k{k}_{stage}"] = variants[stage]

    with Image.open(project_root / record["mask_path"]) as source:
        reference = (
            np.array(source.convert("L")) >= REFERENCE_THRESHOLD
        )

    if reference.shape != image.shape[:2]:
        raise SystemExit(
            f"Dimension mismatch: {record['sample_id']}"
        )

    masks["all_foreground"] = np.ones_like(reference)

    for method, predicted in masks.items():
        dice, iou = overlap_scores(predicted, reference)

        results.append({
            "sample_id": record["sample_id"],
            "category": record["category"],
            "group_id": record["group_id"],
            "split": "val",
            "method": method,
            "dice": dice,
            "iou": iou,
            "predicted_foreground_fraction": float(predicted.mean()),
            "reference_foreground_fraction": float(reference.mean()),
        })

    if number % 50 == 0 or number == len(validation_records):
        print(f"Evaluated {number}/{len(validation_records)} images")


methods = [
    f"k{k}_{stage}"
    for k in (2, 3, 4)
    for stage in STAGES
] + ["all_foreground"]

if len(results) != len(validation_records) * len(methods):
    raise SystemExit("Unexpected number of result rows.")

results_dir = project_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

output_path = results_dir / "kmeans_morph_val.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=list(results[0])
    )
    writer.writeheader()
    writer.writerows(results)


def category_mean(method, category, metric):
    return float(np.mean([
        row[metric]
        for row in results
        if row["method"] == method
        and row["category"] == category
    ]))


print("\nOpenCV:", cv2.__version__)
print("Validation images evaluated:", len(validation_records))

for k in (2, 3, 4):
    print(f"\nk={k}: mean Dice within each category")
    print(
        f"{'Category':<20} {'Images':>7} "
        f"{'M0':>9} {'M1':>9} {'M2':>9} {'All FG':>9}"
    )

    for category in categories:
        values = [
            category_mean(f"k{k}_{stage}", category, "dice")
            for stage in STAGES
        ]
        control = category_mean(
            "all_foreground", category, "dice"
        )

        print(
            f"{category:<20} {category_counts[category]:>7} "
            + " ".join(f"{value:>9.4f}" for value in values)
            + f" {control:>9.4f}"
        )

print("\nEqual-weight mean across categories:")

for method in methods:
    mean_dice = float(np.mean([
        category_mean(method, category, "dice")
        for category in categories
    ]))
    mean_iou = float(np.mean([
        category_mean(method, category, "iou")
        for category in categories
    ]))

    print(
        f"  {method:<16} "
        f"Dice={mean_dice:.4f}, IoU={mean_iou:.4f}"
    )

provenance = {
    "evaluation_subset": "val",
    "image_count": len(validation_records),
    "result_row_count": len(results),
    "reference_threshold": REFERENCE_THRESHOLD,
    "aggregation": (
        "Mean image score within each category, "
        "then equal-weight mean across categories."
    ),
    "split_sha256": expected_checksum,
    "model_sha256": model_checksums,
    "prediction_helper_sha256": helper_checksum,
    "postprocessing_helper_sha256": postprocessing_checksum,
    "variants": {
        "M0": "raw prediction",
        "M1": "retain components >=16 pixels; connectivity=8",
        "M2": (
            "M1, opening then closing; 3x3 cross; "
            "one iteration each; BORDER_REPLICATE"
        ),
    },
    "versions": {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pillow": pillow_version,
    },
}

run_path = results_dir / "kmeans_morph_val_run.json"

with run_path.open("w", encoding="utf-8", newline="\n") as file:
    json.dump(provenance, file, indent=2, allow_nan=False)
    file.write("\n")

print("\nSaved:", output_path)
print("Saved:", run_path)