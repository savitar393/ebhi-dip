"""Thin CLI runner to build data/metadata/manifest.csv from EBHI-SEG raw files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ebhi_dip.paths import RAW_DATASET_REL, find_project_root, resolve_data_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing data/metadata/manifest.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    dataset_root = resolve_data_root(project_root, data_root_override=args.data_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Raw EBHI-SEG dataset directory not found: {dataset_root}")

    output_path = project_root / "data/metadata/manifest.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing '{output_path}' without --overwrite."
        )

    raw_prefix = RAW_DATASET_REL.as_posix()
    rows: list[dict[str, str]] = []
    for category in sorted(dataset_root.iterdir()):
        if not category.is_dir():
            continue
        for image_path in sorted((category / "image").glob("*.png")):
            mask_path = category / "label" / image_path.name
            rows.append(
                {
                    "sample_id": f"{category.name}/{image_path.stem}",
                    "category": category.name,
                    "image_path": f"{raw_prefix}/{category.name}/image/{image_path.name}",
                    "mask_path": f"{raw_prefix}/{category.name}/label/{image_path.name}",
                    "pair_status": "paired" if mask_path.is_file() else "missing_mask",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "category", "image_path", "mask_path", "pair_status"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    paired = sum(row["pair_status"] == "paired" for row in rows)
    print(f"Total image records: {len(rows)} | Paired: {paired} | Missing masks: {len(rows) - paired}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
