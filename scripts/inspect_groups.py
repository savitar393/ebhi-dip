"""Thin CLI runner to summarize filename prefixes and optionally write filename_groups.csv."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

from ebhi_dip.dataset import canonical_group_id
from ebhi_dip.paths import find_project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Write/overwrite data/metadata/filename_groups.csv (read-only summary by default if file exists).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    metadata_dir = project_root / "data/metadata"

    with (metadata_dir / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
        records = [row for row in csv.DictReader(file) if row["pair_status"] == "paired"]

    raw_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    canonical_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        raw_prefix = Path(record["image_path"]).stem.split("-")[0]
        raw_groups[raw_prefix].append(record)
        canonical_groups[canonical_group_id(record["image_path"])].add(raw_prefix)

    summary = [
        {
            "filename_prefix": prefix,
            "image_count": len(members),
            "category_count": len({row["category"] for row in members}),
            "categories": " | ".join(sorted({row["category"] for row in members})),
        }
        for prefix, members in sorted(raw_groups.items())
    ]

    output_path = metadata_dir / "filename_groups.csv"
    if not output_path.exists() or args.overwrite:
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Saved: {output_path}")

    collisions = [sorted(v) for v in canonical_groups.values() if len(v) > 1]
    print(
        f"Paired images: {len(records)} | Raw prefixes: {len(raw_groups)} | "
        f"Uppercase canonical groups: {len(canonical_groups)} | Case collisions: {len(collisions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
