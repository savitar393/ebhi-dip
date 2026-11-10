"""Thin CLI runner to fit train-only SVM classifiers and evaluate on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ebhi_dip.classification import PRIMARY_FEATURE_SETS, run_svm_experiments
from ebhi_dip.features import DEFAULT_FEATURE_METHODS, PRIMARY_FEATURE_METHOD
from ebhi_dip.paths import find_project_root

OUTPUT_DIR_REL = Path("results/classification")
MODELS_DIR_REL = Path("models")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument(
        "--primary-method",
        default=PRIMARY_FEATURE_METHOD,
        help=f"Primary segmentation configuration for Stage 07 (default: {PRIMARY_FEATURE_METHOD}).",
    )
    parser.add_argument(
        "--comparators",
        nargs="+",
        default=list(DEFAULT_FEATURE_METHODS),
        help=f"Segmentation comparators for morphology/combined sensitivity (default: {' '.join(DEFAULT_FEATURE_METHODS)}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Allow overwriting existing classification artifacts under {OUTPUT_DIR_REL.as_posix()}/ and {MODELS_DIR_REL.as_posix()}/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    output_dir = project_root / OUTPUT_DIR_REL
    models_dir = project_root / MODELS_DIR_REL

    expected_model_ids = [
        *(f"svm_{args.primary_method}_{s}" for s in PRIMARY_FEATURE_SETS),
        f"svm_{args.primary_method}_selected",
    ]
    expected_files = [
        output_dir / "svm_grid_results.csv",
        output_dir / "svm_selected_summary.csv",
        output_dir / "svm_per_category_val.csv",
        output_dir / "svm_comparator_sensitivity.csv",
        output_dir / "svm_val_predictions.csv",
        output_dir / "classification_manifest.json",
        *(models_dir / f"{model_id}.json" for model_id in expected_model_ids),
    ]
    if any(path.exists() for path in expected_files) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing SVM classification artifacts without --overwrite."
        )

    outputs = run_svm_experiments(
        project_root,
        primary_method=args.primary_method,
        comparators=args.comparators,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    outputs.grid_results.to_csv(output_dir / "svm_grid_results.csv", index=False, lineterminator="\n")
    outputs.selected_summary.to_csv(
        output_dir / "svm_selected_summary.csv", index=False, lineterminator="\n"
    )
    outputs.per_category_val.to_csv(
        output_dir / "svm_per_category_val.csv", index=False, lineterminator="\n"
    )
    outputs.comparator_sensitivity.to_csv(
        output_dir / "svm_comparator_sensitivity.csv", index=False, lineterminator="\n"
    )
    outputs.val_predictions.to_csv(
        output_dir / "svm_val_predictions.csv", index=False, lineterminator="\n"
    )

    with (output_dir / "classification_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(outputs.manifest, file, indent=2, allow_nan=False)
        file.write("\n")

    for model_id, payload in outputs.model_artifacts.items():
        model_path = models_dir / f"{model_id}.json"
        with model_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")

    sel = outputs.manifest["selected_primary_model"]
    print(
        f"Stage 07 SVM evaluation complete | primary={args.primary_method} | "
        f"selected={sel['feature_set']} ({sel['kernel']}, C={sel['C']}, gamma={sel['gamma']}, cw={sel['class_weight']}) | "
        f"val_macro_f1={sel['val_macro_f1']:.4f} | val_accuracy={sel['val_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
