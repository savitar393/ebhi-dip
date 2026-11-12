"""Thin CLI runner to extract appearance and mask-morphology feature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ebhi_dip.features import (
    APPEARANCE_FEATURE_NAMES,
    COMPARISON_2X3_METHODS,
    DEFAULT_FEATURE_METHODS,
    DEFAULT_FEATURE_SPLITS,
    FEATURE_NAMES,
    PRIMARY_FEATURE_METHOD,
    build_feature_tables,
    feature_manifest,
    feature_manifest_2x3,
    validate_feature_table,
)
from ebhi_dip.paths import find_project_root

OUTPUT_DIR_REL = Path("results/features")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--protocol",
        choices=("2x3", "legacy-v1"),
        default="2x3",
        help="Feature extraction protocol: '2x3' (all six otsu/k4 M0-M2 configs, preserving existing CSVs) or 'legacy-v1' (4 comparators).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Frozen segmentation comparators (defaults to 2x3 methods under protocol=2x3 or legacy 4 under protocol=legacy-v1).",
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
        "--force-rewrite-existing",
        action="store_true",
        help="Under --protocol 2x3, rewrite already-existing v1 CSVs after verifying parity instead of preserving their exact bytes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Allow overwriting existing feature artifacts under {OUTPUT_DIR_REL.as_posix()}/.",
    )
    return parser


def _max_frame_abs_diff(
    stored: pd.DataFrame,
    recomputed: pd.DataFrame,
    feature_cols: tuple[str, ...],
) -> float:
    left = stored.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    right = recomputed.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if not left["sample_id"].equals(right["sample_id"]):
        raise ValueError("Sample ID mismatch while verifying parity against preserved feature CSV.")
    return float(np.max(np.abs(left[list(feature_cols)].to_numpy(dtype=np.float64) - right[list(feature_cols)].to_numpy(dtype=np.float64))))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    output_dir = project_root / OUTPUT_DIR_REL

    methods = (
        list(args.methods)
        if args.methods is not None
        else list(COMPARISON_2X3_METHODS if args.protocol == "2x3" else DEFAULT_FEATURE_METHODS)
    )

    if args.protocol == "legacy-v1":
        expected_files = [
            output_dir / "appearance.csv",
            output_dir / "morphology_diagnostics.csv",
            output_dir / "feature_manifest.json",
            *(output_dir / f"features_{method}.csv" for method in methods),
        ]
        if any(path.exists() for path in expected_files) and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing feature artifacts under '{output_dir}' without --overwrite."
            )

        tables = build_feature_tables(
            project_root,
            methods=methods,
            splits=args.splits,
            data_root_override=args.data_root,
            workers=args.workers,
        )
        validate_feature_table(
            tables.appearance, feature_names=APPEARANCE_FEATURE_NAMES, require_method=False
        )
        for table in tables.combined.values():
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
            f"Saved legacy v1 artifacts under {output_dir}"
        )
        return 0

    # Protocol 2x3: preserve existing v1 CSVs/manifests untouched where parity holds, and emit new 2x3 artifacts.
    diag_2x3_path = output_dir / "morphology_diagnostics_2x3.csv"
    manifest_2x3_path = output_dir / "feature_manifest_2x3.json"
    app_path = output_dir / "appearance.csv"

    target_write_paths = [diag_2x3_path, manifest_2x3_path]
    if args.force_rewrite_existing or not app_path.exists():
        target_write_paths.append(app_path)
    for method in methods:
        m_path = output_dir / f"features_{method}.csv"
        if args.force_rewrite_existing or not m_path.exists():
            target_write_paths.append(m_path)

    if any(path.exists() for path in target_write_paths) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing 2x3 feature artifacts under '{output_dir}' without --overwrite."
        )

    tables = build_feature_tables(
        project_root,
        methods=methods,
        splits=args.splits,
        data_root_override=args.data_root,
        workers=args.workers,
    )

    validate_feature_table(
        tables.appearance, feature_names=APPEARANCE_FEATURE_NAMES, require_method=False
    )
    for table in tables.combined.values():
        validate_feature_table(table)

    output_dir.mkdir(parents=True, exist_ok=True)
    preserved: dict[str, str] = {}
    parity_diffs: dict[str, float] = {}

    if app_path.exists() and not args.force_rewrite_existing:
        stored_app = pd.read_csv(app_path)
        diff = _max_frame_abs_diff(stored_app, tables.appearance, APPEARANCE_FEATURE_NAMES)
        if diff >= 1e-12:
            raise ValueError(f"Preserved appearance.csv parity check failed: max_abs_diff={diff:.3e}")
        preserved["appearance"] = "preserved_byte_identical"
        parity_diffs["appearance"] = diff
    else:
        tables.appearance.to_csv(app_path, index=False, lineterminator="\n")
        preserved["appearance"] = "written"

    for method in tables.methods:
        m_path = output_dir / f"features_{method}.csv"
        if m_path.exists() and not args.force_rewrite_existing:
            stored_m = pd.read_csv(m_path)
            diff = _max_frame_abs_diff(stored_m, tables.combined[method], FEATURE_NAMES)
            if diff >= 1e-12:
                raise ValueError(
                    f"Preserved {m_path.name} parity check failed: max_abs_diff={diff:.3e}"
                )
            preserved[f"features_{method}"] = "preserved_byte_identical"
            parity_diffs[f"features_{method}"] = diff
        else:
            tables.combined[method].to_csv(m_path, index=False, lineterminator="\n")
            preserved[f"features_{method}"] = "written"

    tables.diagnostics.to_csv(diag_2x3_path, index=False, lineterminator="\n")
    manifest = feature_manifest_2x3(
        tables,
        project_root,
        primary_method=args.primary_method,
        preserved_files=preserved,
        parity_max_abs_diff=parity_diffs,
    )
    with manifest_2x3_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")

    print(
        f"Extracted {len(tables.appearance)} samples | protocol=2x3 | methods={','.join(tables.methods)} | "
        f"Saved artifacts under {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
