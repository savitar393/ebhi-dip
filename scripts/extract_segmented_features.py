"""Extract mask-restricted appearance features for the selected segmentation method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ebhi_dip.features import (
    DEFAULT_FEATURE_SPLITS,
    PRIMARY_FEATURE_METHOD,
    SEGMENTED_APPEARANCE_FEATURE_NAMES,
    SEGMENTED_FEATURE_NAMES,
    build_segmented_feature_tables,
    segmented_feature_manifest,
    validate_feature_table,
)
from ebhi_dip.paths import find_project_root

OUTPUT_DIR_REL = Path("results/features")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--method",
        default=PRIMARY_FEATURE_METHOD,
        help=f"Frozen segmentation method used to restrict appearance features (default: {PRIMARY_FEATURE_METHOD}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_FEATURE_SPLITS),
        help=f"Cohort splits to extract (default: {' '.join(DEFAULT_FEATURE_SPLITS)}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for segmented feature extraction (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Allow overwriting existing segmented artifacts under {OUTPUT_DIR_REL.as_posix()}/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    output_dir = project_root / OUTPUT_DIR_REL

    appearance_path = output_dir / f"appearance_segmented_{args.method}.csv"
    combined_path = output_dir / f"features_{args.method}_segmented.csv"
    manifest_path = output_dir / f"feature_manifest_segmented_{args.method}.json"
    expected_files = [appearance_path, combined_path, manifest_path]
    if any(path.exists() for path in expected_files) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing segmented feature artifacts under '{output_dir}' without --overwrite."
        )

    tables = build_segmented_feature_tables(
        project_root,
        method=args.method,
        splits=args.splits,
        data_root_override=args.data_root,
        workers=args.workers,
    )
    validate_feature_table(
        tables.appearance,
        feature_names=SEGMENTED_APPEARANCE_FEATURE_NAMES,
        require_method=False,
    )
    validate_feature_table(tables.combined, feature_names=SEGMENTED_FEATURE_NAMES)

    manifest = segmented_feature_manifest(tables, project_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables.appearance.to_csv(appearance_path, index=False, lineterminator="\n")
    tables.combined.to_csv(combined_path, index=False, lineterminator="\n")
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")

    print(
        f"Extracted {len(tables.appearance)} segmented samples | method={tables.method} | "
        f"Saved artifacts under {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
