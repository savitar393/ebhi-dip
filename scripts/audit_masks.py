"""Thin CLI runner to audit raw grayscale masks and produce data/metadata/mask_audit.csv."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from ebhi_dip.dataset import REFERENCE_THRESHOLD, load_raw_mask
from ebhi_dip.paths import find_project_root, resolve_manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing data/metadata/mask_audit.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    metadata_dir = project_root / "data/metadata"
    output_path = metadata_dir / "mask_audit.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing '{output_path}' without --overwrite."
        )

    with (metadata_dir / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
        records = list(csv.DictReader(file))

    results: list[dict[str, object]] = []
    for record in records:
        if record["pair_status"] != "paired":
            continue
        mask_path = resolve_manifest_path(
            project_root, record["mask_path"], data_root_override=args.data_root
        )
        mask = load_raw_mask(mask_path)
        gray_pixels = (mask > 0) & (mask < 255)
        binary_mask = mask >= REFERENCE_THRESHOLD
        results.append(
            {
                "sample_id": record["sample_id"],
                "category": record["category"],
                "minimum": int(mask.min()),
                "maximum": int(mask.max()),
                "unique_values": int(np.unique(mask).size),
                "gray_pixels": int(np.count_nonzero(gray_pixels)),
                "gray_percentage": float(100.0 * gray_pixels.mean()),
                "foreground_pixels": int(np.count_nonzero(binary_mask)),
                "foreground_percentage": float(100.0 * binary_mask.mean()),
            }
        )

    if not results:
        raise RuntimeError("No paired masks found in manifest.csv.")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    full_count = sum(int(row["foreground_pixels"]) == 224 * 224 for row in results)
    cat_counts = Counter(
        str(row["category"]) for row in results if int(row["foreground_pixels"]) == 224 * 224
    )
    print(f"Masks audited: {len(results)} | Full-foreground masks: {full_count} ({dict(sorted(cat_counts.items()))})")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
