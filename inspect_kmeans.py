import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from kmeans_prediction import nearest_clusters


TARGET_SAMPLE = "Normal/GT2000000-1-400-001"
REFERENCE_THRESHOLD = 128

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
    record = next(
        (
            row for row in csv.DictReader(file)
            if row["sample_id"] == TARGET_SAMPLE
        ),
        None,
    )

if record is None or record["split"] != "train":
    raise SystemExit("Expected the selected sample in the training split.")

with Image.open(project_root / record["image_path"]) as source:
    image = np.array(source.convert("RGB"))

rgb_float = image.astype(np.float32) / np.float32(255.0)
lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)

# Include raw Otsu as a familiar comparison.
otsu_config = json.loads(
    (project_root / "configs" / "otsu_raw.json")
    .read_text(encoding="utf-8")
)

if (
    otsu_config["split_sha256"] != expected_checksum
    or otsu_config["postprocessing"] != "none"
):
    raise SystemExit("Unexpected raw Otsu configuration.")

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

threshold, bright_u8 = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU,
)

polarity = otsu_config["foreground_polarity"]

if polarity == "dark":
    otsu_mask = bright_u8 == 0
elif polarity == "bright":
    otsu_mask = bright_u8 > 0
else:
    raise SystemExit("Unsupported Otsu polarity.")

predictions = {"Otsu M0": otsu_mask}

helper_checksum = hashlib.sha256(
    (project_root / "kmeans_prediction.py").read_bytes()
).hexdigest()

for k in (2, 3, 4):
    model_path = (
        project_root / "models" / f"kmeans_lab_k{k}_mapped.json"
    )
    model = json.loads(model_path.read_text(encoding="utf-8"))

    if (
        model["split_sha256"] != expected_checksum
        or model["prediction"]["helper_sha256"] != helper_checksum
        or model["foreground_mapping"]["reference_threshold"]
        != REFERENCE_THRESHOLD
    ):
        raise SystemExit(f"Configuration mismatch for k={k}.")

    cluster_labels = nearest_clusters(
        lab.reshape(-1, 3),
        model["cluster_centers"],
    ).reshape(image.shape[:2])

    predictions[f"K-means k={k}"] = np.isin(
        cluster_labels,
        model["foreground_clusters"],
    )

# The reference is used here to score predictions.
with Image.open(project_root / record["mask_path"]) as source:
    reference = (
        np.array(source.convert("L")) >= REFERENCE_THRESHOLD
    )

if reference.shape != image.shape[:2]:
    raise SystemExit("Image and reference dimensions differ.")


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


scored_masks = {
    **predictions,
    "All foreground": np.ones_like(reference),
}

scores = {
    name: overlap_scores(mask, reference)
    for name, mask in scored_masks.items()
}

print("Sample:", record["sample_id"])
print("Subset:", record["split"])
print(f"Reference foreground: {100 * reference.mean():.2f}%")
print(f"\n{'Method':<20} {'Dice':>8} {'IoU':>8} {'FG %':>8}")

for name, mask in scored_masks.items():
    dice, iou = scores[name]
    print(
        f"{name:<20} {dice:>8.4f} {iou:>8.4f} "
        f"{100 * mask.mean():>8.2f}"
    )

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(image, interpolation="nearest")
axes[0, 0].set_title("Original RGB")

axes[0, 1].imshow(
    reference, cmap="gray", vmin=0, vmax=1,
    interpolation="nearest",
)
axes[0, 1].set_title("Binary reference: mask >= 128")

prediction_axes = [
    axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]
]

for axis, (name, mask) in zip(
    prediction_axes, predictions.items()
):
    dice, iou = scores[name]

    axis.imshow(
        mask, cmap="gray", vmin=0, vmax=1,
        interpolation="nearest",
    )
    axis.set_title(
        f"{name}\nDice={dice:.4f}, IoU={iou:.4f}"
    )

for axis in axes.ravel():
    axis.axis("off")

fig.suptitle(f"{record['sample_id']} | train")
plt.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)
plt.show()