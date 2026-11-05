"""Thin CLI runner to generate provisional grouped train/val/test candidate split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ebhi_dip.dataset import ALLOWED_SPLITS, canonical_group_id
from ebhi_dip.paths import find_project_root

SEED = 42
CANDIDATES = 10_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting data/metadata/split_candidate.csv (never touches frozen split).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    metadata_dir = project_root / "data/metadata"
    output_path = metadata_dir / "split_candidate.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing '{output_path}' without --overwrite."
        )

    with (metadata_dir / "image_hashes.csv").open(newline="", encoding="utf-8-sig") as file:
        records = [row for row in csv.DictReader(file) if row["pair_status"] == "paired"]

    for row in records:
        row["group_id"] = canonical_group_id(row["image_path"])

    categories = sorted({row["category"] for row in records})
    group_names = sorted({row["group_id"] for row in records})
    cat_idx = {name: i for i, name in enumerate(categories)}
    grp_idx = {name: i for i, name in enumerate(group_names)}

    group_counts = np.zeros((len(group_names), len(categories)), dtype=np.int64)
    for row in records:
        group_counts[grp_idx[row["group_id"]], cat_idx[row["category"]]] += 1

    category_totals = group_counts.sum(axis=0)
    targets = np.array([0.70, 0.15, 0.15])
    n_train = round(0.70 * len(group_names))
    n_val = round(0.15 * len(group_names))

    rng = np.random.default_rng(SEED)
    best_score = float("inf")
    best_parts = None
    for _ in range(CANDIDATES):
        order = rng.permutation(len(group_names))
        parts = np.split(order, [n_train, n_train + n_val])
        counts = np.stack([group_counts[indices].sum(axis=0) for indices in parts])
        if np.any(counts == 0):
            continue
        score = float(np.mean((counts / category_totals - targets[:, None]) ** 2))
        if score < best_score:
            best_score = score
            best_parts = [indices.copy() for indices in parts]

    if best_parts is None:
        raise RuntimeError("No candidate split contained all categories in every subset.")

    assignment = {
        group_names[idx]: split_name
        for split_name, indices in zip(ALLOWED_SPLITS, best_parts)
        for idx in indices
    }
    for row in records:
        row["split"] = assignment[row["group_id"]]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved candidate split ({len(records)} samples, {len(group_names)} groups): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
