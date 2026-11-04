import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SEED = 42
PIXELS_PER_IMAGE = 1000
REFERENCE_THRESHOLD = 128

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
    training_records = sorted(
        (
            row for row in csv.DictReader(file)
            if row["split"] == "train"
        ),
        key=lambda row: row["sample_id"],
    )

if not training_records:
    raise SystemExit("No training records found.")

rng = np.random.default_rng(SEED)

lab_parts = []
reference_parts = []
index_parts = []

for record in training_records:
    with Image.open(project_root / record["image_path"]) as source:
        image = np.array(source.convert("RGB"))

    with Image.open(project_root / record["mask_path"]) as source:
        mask = np.array(source.convert("L"))

    if image.shape != (224, 224, 3) or mask.shape != (224, 224):
        raise SystemExit(
            f"Unexpected dimensions: {record['sample_id']}"
        )

    rgb_float = image.astype(np.float32) / np.float32(255.0)
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)

    pixel_table = lab.reshape(-1, 3)
    reference_table = mask.reshape(-1) >= REFERENCE_THRESHOLD

    indices = rng.choice(
        pixel_table.shape[0],
        size=PIXELS_PER_IMAGE,
        replace=False,
    )

    # Use exactly the same positions for colors and reference labels.
    lab_parts.append(pixel_table[indices])
    reference_parts.append(reference_table[indices])
    index_parts.append(indices.astype(np.uint32))

X = np.concatenate(lab_parts, axis=0)
y = np.concatenate(reference_parts, axis=0)

if not np.isfinite(X).all():
    raise SystemExit("Sampled Lab coordinates contain nonfinite values.")

output_dir = project_root / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "kmeans_train_pixels.npz"

np.savez_compressed(
    output_path,
    lab_pixels=X,
    reference_foreground=y,
    sample_ids=np.array([
        record["sample_id"] for record in training_records
    ]),
    sampled_pixel_indices=np.stack(index_parts),
    pixels_per_image=PIXELS_PER_IMAGE,
    reference_threshold=REFERENCE_THRESHOLD,
    seed=SEED,
    subset="train",
    sample_order="sample_id ascending",
    color_conversion="RGB float32 / 255 -> COLOR_RGB2LAB",
    split_sha256=expected_checksum,
    opencv_version=cv2.__version__,
    numpy_version=np.__version__,
)

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Training images sampled:", len(training_records))
print("Pixels per image:", PIXELS_PER_IMAGE)
print("Random seed:", SEED)
print("Lab sample shape:", X.shape)
print("Lab sample dtype:", X.dtype)
print("Reference label shape:", y.shape)
print("Reference label dtype:", y.dtype)
print(f"Sampled reference foreground: {100 * y.mean():.2f}%")
print("Saved:", output_path)