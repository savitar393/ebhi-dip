import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"
split_path = metadata_dir / "split_manifest.csv"
cache_path = (
    project_root / "data" / "processed" / "kmeans_train_pixels.npz"
)

expected_checksum = (
    (metadata_dir / "split_manifest.sha256")
    .read_text(encoding="utf-8")
    .split()[0]
)

assert (
    hashlib.sha256(split_path.read_bytes()).hexdigest()
    == expected_checksum
), "Frozen split checksum mismatch."

with split_path.open(newline="", encoding="utf-8-sig") as file:
    training_records = sorted(
        (
            row for row in csv.DictReader(file)
            if row["split"] == "train"
        ),
        key=lambda row: row["sample_id"],
    )

with np.load(cache_path, allow_pickle=False) as cache:
    X = cache["lab_pixels"]
    y = cache["reference_foreground"]
    sample_ids = cache["sample_ids"].tolist()
    indices = cache["sampled_pixel_indices"]

    pixels_per_image = int(cache["pixels_per_image"].item())
    reference_threshold = int(cache["reference_threshold"].item())

    assert cache["split_sha256"].item() == expected_checksum
    assert cache["subset"].item() == "train"
    assert int(cache["seed"].item()) == 42

assert pixels_per_image == 1000
assert reference_threshold == 128

assert sample_ids == [
    record["sample_id"] for record in training_records
], "Cached image IDs or ordering differ from the training split."

expected_rows = len(training_records) * pixels_per_image

assert X.shape == (expected_rows, 3)
assert y.shape == (expected_rows,)
assert X.dtype == np.float32
assert y.dtype == np.bool_
assert np.isfinite(X).all()

assert indices.shape == (len(training_records), pixels_per_image)
assert np.all((indices >= 0) & (indices < 224 * 224))

# Sorted positions must strictly increase within each image.
assert np.all(
    np.diff(np.sort(indices, axis=1), axis=1) > 0
), "Repeated sampled positions within an image."

# Independently reload the first source image and its reference.
record = training_records[0]

with Image.open(project_root / record["image_path"]) as source:
    image = np.array(source.convert("RGB"))

with Image.open(project_root / record["mask_path"]) as source:
    mask = np.array(source.convert("L"))

rgb_float = image.astype(np.float32) / np.float32(255.0)
lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)

source_lab = lab.reshape(-1, 3)[indices[0]]
source_labels = (
    mask.reshape(-1)[indices[0]] >= reference_threshold
)

assert np.array_equal(
    X[:pixels_per_image], source_lab
), "Cached Lab coordinates differ from the sampled source pixels."

assert np.array_equal(
    y[:pixels_per_image], source_labels
), "Cached labels differ from the sampled reference pixels."

print("Training image IDs and order: verified")
print("Lab sample shape:", X.shape)
print("Reference label shape:", y.shape)
print("Finite Lab coordinates: verified")
print("Distinct sampled positions within every image: verified")
print("Source alignment checked for:", record["sample_id"])
print("  Lab coordinates: exact match")
print("  Reference labels: exact match")