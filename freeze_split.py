import csv
import hashlib
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def require(condition, message):
    if not condition:
        raise SystemExit(f"Check failed: {message}")


manifest = read_csv(metadata_dir / "manifest.csv")
candidate_path = metadata_dir / "split_candidate.csv"
candidate_bytes = candidate_path.read_bytes()
candidate = read_csv(candidate_path)

expected = {
    row["sample_id"]: row
    for row in manifest
    if row["pair_status"] == "paired"
}

sample_ids = [row["sample_id"] for row in candidate]

require(
    len(sample_ids) == len(set(sample_ids)),
    "Repeated sample IDs in the candidate.",
)
require(
    set(sample_ids) == set(expected),
    "Candidate does not contain exactly the paired samples.",
)

allowed_splits = {"train", "val", "test"}
group_splits = defaultdict(set)
pixel_splits = defaultdict(set)
categories_by_split = defaultdict(set)

for row in candidate:
    original = expected[row["sample_id"]]

    for field in ("category", "image_path", "mask_path", "pair_status"):
        require(
            row[field] == original[field],
            f"{field} changed for {row['sample_id']}.",
        )

    expected_group = (
        Path(row["image_path"]).stem.split("-")[0].upper()
    )

    require(
        row["group_id"] == expected_group,
        f"Unexpected group ID for {row['sample_id']}.",
    )
    require(
        row["split"] in allowed_splits,
        f"Unknown subset for {row['sample_id']}.",
    )
    require(
        bool(row["pixel_sha256"]),
        f"Missing pixel fingerprint for {row['sample_id']}.",
    )

    group_splits[row["group_id"]].add(row["split"])
    pixel_splits[row["pixel_sha256"]].add(row["split"])
    categories_by_split[row["split"]].add(row["category"])

require(
    all(len(subsets) == 1 for subsets in group_splits.values()),
    "A filename group crosses subsets.",
)
require(
    all(len(subsets) == 1 for subsets in pixel_splits.values()),
    "A recorded image fingerprint crosses subsets.",
)

all_categories = {row["category"] for row in expected.values()}

for split_name in sorted(allowed_splits):
    require(
        categories_by_split[split_name] == all_categories,
        f"A category is missing from {split_name}.",
    )

frozen_path = metadata_dir / "split_manifest.csv"
checksum_path = metadata_dir / "split_manifest.sha256"

require(
    not frozen_path.exists() and not checksum_path.exists(),
    "Frozen split files already exist; they were left unchanged.",
)

checksum = hashlib.sha256(candidate_bytes).hexdigest()

with frozen_path.open("xb") as file:
    file.write(candidate_bytes)

with checksum_path.open("x", encoding="utf-8") as file:
    file.write(f"{checksum}  {frozen_path.name}\n")

print("Verified samples:", len(candidate))
print("Verified filename groups:", len(group_splits))
print("All categories present in every subset.")
print("No filename-group or recorded-fingerprint overlap.")
print("Saved:", frozen_path)
print("SHA-256:", checksum)