from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import csv

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"

with (metadata_dir / "mask_audit.csv").open(
    newline="", encoding="utf-8-sig"
) as file:
    target = max(
        csv.DictReader(file),
        key=lambda row: float(row["gray_percentage"]),
    )

with (metadata_dir / "manifest.csv").open(
    newline="", encoding="utf-8-sig"
) as file:
    record = next(
        row for row in csv.DictReader(file)
        if row["sample_id"] == target["sample_id"]
    )

image_path = project_root / record["image_path"]
mask_path = project_root / record["mask_path"]

print("Category:", record["category"])
print("Sample:", record["sample_id"])
print("Audit gray percentage:", target["gray_percentage"])
print("Audit foreground percentage:", target["foreground_percentage"])

with Image.open(image_path) as source:
    print("Image mode:", source.mode)
    image = np.array(source)

with Image.open(mask_path) as source:
    print("Mask mode:", source.mode)
    mask = np.array(source)

print("Filename:", image_path.name)
print("Image shape:", image.shape)
print("Image dtype:", image.dtype)
print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)
print("Mask minimum:", mask.min())
print("Mask maximum:", mask.max())
print("Number of unique mask values:", np.unique(mask).size)
gray_pixels = (mask > 0) & (mask < 255)

print("Black pixels:", np.count_nonzero(mask == 0))
print("White pixels:", np.count_nonzero(mask == 255))
print("Intermediate gray pixels:", np.count_nonzero(gray_pixels))
print(f"Intermediate gray percentage: {100 * gray_pixels.mean():.2f}%")

binary_mask = mask >= 128

print("Binary mask dtype:", binary_mask.dtype)
print("Binary mask values:", np.unique(binary_mask))
print("Foreground pixels:", np.count_nonzero(binary_mask))
print(f"Foreground coverage: {100 * binary_mask.mean():.2f}%")

overlay = image.copy()
red = np.array([255, 0, 0])

overlay[binary_mask] = (
    0.65 * image[binary_mask] + 0.35 * red
).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image)
axes[0].set_title("Original image")

axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255,
               interpolation="nearest")
axes[1].set_title("Reference mask")

axes[2].imshow(overlay, interpolation="nearest")
axes[2].set_title("Reference foreground overlay")

for axis in axes:
    axis.axis("off")

fig.suptitle(image_path.name)
plt.tight_layout()
plt.show()