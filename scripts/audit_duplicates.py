"""Thin CLI runner to hash decoded RGB pixels and produce data/metadata/image_hashes.csv."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

from ebhi_dip.dataset import compute_pixel_sha256, load_rgb
from ebhi_dip.paths import find_project_root, resolve_manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing data/metadata/image_hashes.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    metadata_dir = project_root / "data/metadata"
    output_path = metadata_dir / "image_hashes.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing '{output_path}' without --overwrite."
        )

    with (metadata_dir / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
        records = list(csv.DictReader(file))

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    results: list[dict[str, str]] = []
    for record in records:
        image_path = resolve_manifest_path(
            project_root, record["image_path"], data_root_override=args.data_root
        )
        pixel_hash = compute_pixel_sha256(load_rgb(image_path))
        groups[pixel_hash].append(record)
        results.append({**record, "pixel_sha256": pixel_hash})

    duplicate_groups = [members for members in groups.values() if len(members) > 1]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(
        f"Images scanned: {len(results)} | Distinct fingerprints: {len(groups)} | "
        f"Duplicate groups: {len(duplicate_groups)}"
    )
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
