import csv
import hashlib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"
split_path = metadata_dir / "split_manifest.csv"

expected_checksum = (
    (metadata_dir / "split_manifest.sha256")
    .read_text(encoding="utf-8")
    .split()[0]
)

actual_checksum = hashlib.sha256(split_path.read_bytes()).hexdigest()

if actual_checksum != expected_checksum:
    raise SystemExit("Frozen split checksum does not match.")

with split_path.open(newline="", encoding="utf-8-sig") as file:
    record = next(
        row for row in csv.DictReader(file)
        if row["split"] == "train" and row["category"] == "Normal"
    )

with Image.open(project_root / record["image_path"]) as source:
    image = np.array(source.convert("RGB"))

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

threshold, bright_u8 = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU,
)

bright_mask = bright_u8 > 0
dark_mask = ~bright_mask

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Sample:", record["sample_id"])
print("Subset:", record["split"])
print("Grayscale shape:", gray.shape)
print("Grayscale dtype:", gray.dtype)
print("Otsu threshold:", threshold)
print(f"Dark-side coverage: {100 * dark_mask.mean():.2f}%")
print(f"Bright-side coverage: {100 * bright_mask.mean():.2f}%")

with Image.open(project_root / record["mask_path"]) as source:
    reference = np.array(source) >= 128


def overlap_scores(predicted, reference):
    intersection = np.count_nonzero(predicted & reference)
    union = np.count_nonzero(predicted | reference)
    total = (
        np.count_nonzero(predicted)
        + np.count_nonzero(reference)
    )

    # Convention: two empty masks have perfect overlap.
    dice = 2 * intersection / total if total else 1.0
    iou = intersection / union if union else 1.0

    return dice, iou


print(f"Reference foreground coverage: {100 * reference.mean():.2f}%")

for name, predicted in [
    ("Dark side", dark_mask),
    ("Bright side", bright_mask),
]:
    dice, iou = overlap_scores(predicted, reference)
    print(f"{name}: Dice={dice:.4f}, IoU={iou:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Original RGB")

axes[0, 1].imshow(
    reference, cmap="gray", vmin=0, vmax=1,
    interpolation="nearest",
)
axes[0, 1].set_title("Binary reference: mask >= 128")

axes[1, 0].imshow(
    dark_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
)
axes[1, 0].set_title(f"Dark side: intensity <= {threshold:g}")

axes[1, 1].imshow(
    bright_mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
)
axes[1, 1].set_title(f"Bright side: intensity > {threshold:g}")

for axis in axes.flat:
    axis.axis("off")

fig.suptitle(record["sample_id"])
plt.tight_layout()
plt.show()