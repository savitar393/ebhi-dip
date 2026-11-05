"""Thin CLI runner to verify or freeze split_candidate.csv into split_manifest.csv."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path

from ebhi_dip.dataset import ALLOWED_SPLITS, canonical_group_id, load_split_manifest
from ebhi_dip.paths import FROZEN_SPLIT_REL, FROZEN_SPLIT_SHA_REL, find_project_root, verify_frozen_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing frozen split bytes and cohort invariants without writing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)

    frozen_path = project_root / FROZEN_SPLIT_REL
    checksum_path = project_root / FROZEN_SPLIT_SHA_REL

    if args.verify_only or (frozen_path.exists() and checksum_path.exists()):
        if not args.verify_only:
            raise FileExistsError(
                "Frozen split files already exist and are immutable; pass --verify-only to audit."
            )
        digest = verify_frozen_split(project_root)
        split_df = load_split_manifest(project_root, verify=False)
        print(
            f"Verified frozen split: {len(split_df)} samples, "
            f"{split_df['group_id'].nunique()} groups, SHA-256={digest}"
        )
        return 0

    metadata_dir = project_root / "data/metadata"
    candidate_path = metadata_dir / "split_candidate.csv"
    candidate_bytes = candidate_path.read_bytes()

    with (metadata_dir / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
        expected = {
            row["sample_id"]: row for row in csv.DictReader(file) if row["pair_status"] == "paired"
        }
    with candidate_path.open(newline="", encoding="utf-8-sig") as file:
        candidate = list(csv.DictReader(file))

    sample_ids = [row["sample_id"] for row in candidate]
    if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != set(expected):
        raise ValueError("Candidate split sample_id set does not match paired manifest.")

    group_splits: dict[str, set[str]] = defaultdict(set)
    pixel_splits: dict[str, set[str]] = defaultdict(set)
    categories_by_split: dict[str, set[str]] = defaultdict(set)
    for row in candidate:
        orig = expected[row["sample_id"]]
        for field in ("category", "image_path", "mask_path", "pair_status"):
            if row[field] != orig[field]:
                raise ValueError(f"Field '{field}' differs from manifest for {row['sample_id']}.")
        if row["group_id"] != canonical_group_id(row["image_path"]):
            raise ValueError(f"Invalid canonical group_id for {row['sample_id']}.")
        if row["split"] not in ALLOWED_SPLITS or not row["pixel_sha256"]:
            raise ValueError(f"Invalid split or pixel_sha256 for {row['sample_id']}.")
        group_splits[row["group_id"]].add(row["split"])
        pixel_splits[row["pixel_sha256"]].add(row["split"])
        categories_by_split[row["split"]].add(row["category"])

    if any(len(s) != 1 for s in group_splits.values()):
        raise ValueError("Group overlap across subsets detected.")
    if any(len(s) != 1 for s in pixel_splits.values()):
        raise ValueError("Pixel fingerprint overlap across subsets detected.")

    all_cats = {row["category"] for row in expected.values()}
    if any(categories_by_split[sp] != all_cats for sp in ALLOWED_SPLITS):
        raise ValueError("Missing category in at least one subset.")

    checksum = hashlib.sha256(candidate_bytes).hexdigest()
    with frozen_path.open("xb") as file:
        file.write(candidate_bytes)
    with checksum_path.open("x", encoding="utf-8") as file:
        file.write(f"{checksum}  {frozen_path.name}\n")

    print(f"Saved frozen split: {frozen_path} (SHA-256: {checksum})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
