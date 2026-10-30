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


m0 = dark_mask

# M1: keep foreground components containing at least 16 pixels.
count, labels, stats, _ = cv2.connectedComponentsWithStats(
    m0.astype(np.uint8),
    connectivity=8,
)

keep = stats[:, cv2.CC_STAT_AREA] >= 16
keep[0] = False  # Label 0 represents background.
m1 = keep[labels]

# A radius-1 disk on the pixel grid.
kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
], dtype=np.uint8)

# M2: opening, then closing.
opened = cv2.morphologyEx(
    m1.astype(np.uint8),
    cv2.MORPH_OPEN,
    kernel,
    iterations=1,
    borderType=cv2.BORDER_REPLICATE,
)

m2 = cv2.morphologyEx(
    opened,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1,
    borderType=cv2.BORDER_REPLICATE,
).astype(bool)

print("\nM0 foreground components:", count - 1)
print("M1 retained components:", int(keep.sum()))

for name, predicted in [("M0", m0), ("M1", m1), ("M2", m2)]:
    dice, iou = overlap_scores(predicted, reference)
    print(
        f"{name}: Dice={dice:.4f}, IoU={iou:.4f}, "
        f"coverage={100 * predicted.mean():.2f}%"
    )

images = [
    image,
    gray,
    reference.astype(np.uint8) * 255,
    m0.astype(np.uint8) * 255,
    m1.astype(np.uint8) * 255,
    m2.astype(np.uint8) * 255,
]

titles = [
    "Original RGB",
    "Grayscale",
    "Binary reference",
    "M0: raw Otsu",
    "M1: remove small components",
    "M2: M1 + opening + closing",
]

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

for axis, displayed, title in zip(axes.flat, images, titles):
    axis.imshow(
        displayed, cmap="gray", vmin=0, vmax=255,
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.axis("off")

fig.suptitle(record["sample_id"])
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()