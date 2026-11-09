"""Thin CLI runner to extract appearance and mask-morphology feature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ebhi_dip.features import (
    APPEARANCE_FEATURE_NAMES,
    DEFAULT_FEATURE_METHODS,
    DEFAULT_FEATURE_SPLITS,
    PRIMARY_FEATURE_METHOD,
    build_feature_tables,
    feature_manifest,
    validate_feature_table,
)
from ebhi_dip.paths import find_project_root

OUTPUT_DIR_REL = Path("results/features")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_FEATURE_METHODS),
        help=f"Frozen segmentation comparators (default: {' '.join(DEFAULT_FEATURE_METHODS)}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_FEATURE_SPLITS),
        help=f"Cohort splits to extract (default: {' '.join(DEFAULT_FEATURE_SPLITS)}).",
    )
    parser.add_argument(
        "--primary-method",
        default=PRIMARY_FEATURE_METHOD,
        help=f"Selected segmentation configuration recorded in the manifest (default: {PRIMARY_FEATURE_METHOD}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for feature extraction (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Allow overwriting existing artifacts under {OUTPUT_DIR_REL.as_posix()}/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    output_dir = project_root / OUTPUT_DIR_REL

    expected_files = [
        output_dir / "appearance.csv",
        output_dir / "morphology_diagnostics.csv",
        output_dir / "feature_manifest.json",
        *(output_dir / f"features_{method}.csv" for method in args.methods),
    ]
    if any(path.exists() for path in expected_files) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing feature artifacts under '{output_dir}' without --overwrite."
        )

    tables = build_feature_tables(
        project_root,
        methods=args.methods,
        splits=args.splits,
        data_root_override=args.data_root,
        workers=args.workers,
    )

    validate_feature_table(
        tables.appearance, feature_names=APPEARANCE_FEATURE_NAMES, require_method=False
    )
    for method, table in tables.combined.items():
        validate_feature_table(table)

    manifest = feature_manifest(tables, project_root, primary_method=args.primary_method)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables.appearance.to_csv(expected_files[0], index=False, lineterminator="\n")
    for method in tables.methods:
        tables.combined[method].to_csv(
            output_dir / f"features_{method}.csv",
            index=False,
            lineterminator="\n",
        )
    tables.diagnostics.to_csv(expected_files[1], index=False, lineterminator="\n")
    with expected_files[2].open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")

    print(
        f"Extracted {len(tables.appearance)} samples | methods={','.join(tables.methods)} | "
        f"Saved artifacts under {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
