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

if hashlib.sha256(split_path.read_bytes()).hexdigest() != expected_checksum:
    raise SystemExit("Frozen split checksum does not match.")

with split_path.open(newline="", encoding="utf-8-sig") as file:
    record = next(
        row for row in csv.DictReader(file)
        if row["split"] == "train" and row["category"] == "Normal"
    )

with Image.open(project_root / record["image_path"]) as source:
    image = np.array(source.convert("RGB"))

rgb_float = image.astype(np.float32) / np.float32(255.0)
lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)

pixels = lab.reshape(-1, 3)

print("Sample:", record["sample_id"])
print("Lab shape:", lab.shape)
print("Lab dtype:", lab.dtype)
print("Pixel table shape:", pixels.shape)
print("First pixel RGB:", image[0, 0])
print("First pixel Lab:", pixels[0])

channel_names = ("L*", "a*", "b*")

for index, name in enumerate(channel_names):
    channel = lab[:, :, index]
    print(
        f"{name}: minimum={channel.min():.3f}, "
        f"maximum={channel.max():.3f}, "
        f"mean={channel.mean():.3f}"
    )

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

axes[0].imshow(image)
axes[0].set_title("Original RGB")

for axis, index, name in zip(axes[1:], range(3), channel_names):
    displayed = axis.imshow(
        lab[:, :, index],
        cmap="gray",
        interpolation="nearest",
    )
    axis.set_title(f"Lab {name}")
    fig.colorbar(displayed, ax=axis, shrink=0.8)

for axis in axes:
    axis.axis("off")

fig.suptitle(record["sample_id"])
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()