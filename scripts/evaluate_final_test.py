"""Thin CLI runner for one-time Stage 08 held-out test evaluation (segmentation + six-class SVM)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ebhi_dip.evaluation import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    SEGMENTATION_SELECTION_REL,
    TEST_RESULTS_DIR_REL,
    run_final_test_evaluation,
)
from ebhi_dip.paths import find_project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help=f"Group-blocked bootstrap resamples over the 53 test groups (default: {DEFAULT_BOOTSTRAP_RESAMPLES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"Random seed for deterministic group bootstrap (default: {DEFAULT_BOOTSTRAP_SEED}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            f"Allow overwriting existing Stage 08 artifacts under "
            f"{SEGMENTATION_SELECTION_REL.as_posix()} and {TEST_RESULTS_DIR_REL.as_posix()}/."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    selection_path = project_root / SEGMENTATION_SELECTION_REL
    output_dir = project_root / TEST_RESULTS_DIR_REL

    expected_files = [
        selection_path,
        output_dir / "segmentation_test_per_image.csv",
        output_dir / "segmentation_test_macro_summary.csv",
        output_dir / "segmentation_test_per_category.csv",
        output_dir / "segmentation_test_paired_contrasts.csv",
        output_dir / "classification_test_summary.csv",
        output_dir / "classification_test_per_category.csv",
        output_dir / "classification_test_paired_contrasts.csv",
        output_dir / "classification_test_predictions.csv",
        output_dir / "final_test_manifest.json",
    ]
    if any(path.exists() for path in expected_files) and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing Stage 08 held-out test artifacts without --overwrite."
        )

    outputs = run_final_test_evaluation(
        project_root,
        data_root_override=args.data_root,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )

    selection_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with selection_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(outputs.selection_config, file, indent=2, allow_nan=False)
        file.write("\n")

    outputs.segmentation.per_image.to_csv(
        output_dir / "segmentation_test_per_image.csv", index=False, lineterminator="\n"
    )
    outputs.segmentation.macro_summary.to_csv(
        output_dir / "segmentation_test_macro_summary.csv", index=False, lineterminator="\n"
    )
    outputs.segmentation.per_category.to_csv(
        output_dir / "segmentation_test_per_category.csv", index=False, lineterminator="\n"
    )
    outputs.segmentation.paired_contrasts.to_csv(
        output_dir / "segmentation_test_paired_contrasts.csv", index=False, lineterminator="\n"
    )

    outputs.classification.summary.to_csv(
        output_dir / "classification_test_summary.csv", index=False, lineterminator="\n"
    )
    outputs.classification.per_category.to_csv(
        output_dir / "classification_test_per_category.csv", index=False, lineterminator="\n"
    )
    outputs.classification.paired_contrasts.to_csv(
        output_dir / "classification_test_paired_contrasts.csv", index=False, lineterminator="\n"
    )
    outputs.classification.predictions.to_csv(
        output_dir / "classification_test_predictions.csv", index=False, lineterminator="\n"
    )

    with (output_dir / "final_test_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(outputs.manifest, file, indent=2, allow_nan=False)
        file.write("\n")

    seg_res = outputs.manifest["primary_segmentation_test_result"]
    cls_res = outputs.manifest["primary_classification_test_result"]
    comp_res = outputs.manifest["best_comparator_classification_test_result"]
    print(
        f"Stage 08 final test evaluation complete (319 images, 53 groups) | "
        f"seg k4_M0 test_dice={seg_res['test_mean_category_dice']:.4f} | "
        f"cls k4_M0 Set C test_macro_f1={cls_res['test_macro_f1']:.4f} (acc={cls_res['test_accuracy']:.4f}) | "
        f"cls otsu_M2 Set C test_macro_f1={comp_res['test_macro_f1']:.4f} (acc={comp_res['test_accuracy']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
