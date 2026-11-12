"""Thin CLI runner to fit train-only SVM classifiers and evaluate on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ebhi_dip.classification import (
    CLASSIFICATION_2X3_DIR_REL,
    MODELS_CV_2X3_DIR_REL,
    PRIMARY_FEATURE_SETS,
    run_svm_2x3_experiments,
    run_svm_experiments,
)
from ebhi_dip.features import (
    COMPARISON_2X3_METHODS,
    DEFAULT_FEATURE_METHODS,
    PRIMARY_FEATURE_METHOD,
)
from ebhi_dip.paths import find_project_root
from ebhi_dip.strict_cv import (
    CLASSIFICATION_2X3_STRICT_DIR_REL,
    MODELS_CV_2X3_STRICT_DIR_REL,
    run_svm_2x3_strict_experiments,
    save_strict_2x3_outputs,
)

OUTPUT_DIR_REL = Path("results/classification")
MODELS_DIR_REL = Path("models")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument(
        "--protocol",
        choices=("2x3-strict", "2x3", "legacy-v1"),
        default="2x3-strict",
        help=(
            "Evaluation protocol: '2x3-strict' (strict fold-aware upstream CV & val-blind "
            "selection in results/classification_2x3_strict/, default), '2x3' (fixed-upstream CV in "
            "results/classification_2x3/), or 'legacy-v1'."
        ),
    )
    parser.add_argument(
        "--primary-method",
        default=PRIMARY_FEATURE_METHOD,
        help=f"Primary segmentation reference method for A_seg/C_seg (default: {PRIMARY_FEATURE_METHOD}).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Segmentation methods under protocol=2x3/2x3-strict (default: 6 configurations otsu/k4 M0-M2).",
    )
    parser.add_argument(
        "--comparators",
        nargs="+",
        default=list(DEFAULT_FEATURE_METHODS),
        help=f"Segmentation comparators under protocol=legacy-v1 (default: {' '.join(DEFAULT_FEATURE_METHODS)}).",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker count for fold feature extraction.")
    parser.add_argument("--no-cache", action="store_true", help="Disable strict fold feature cache reuse.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing classification artifacts in the target output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)

    if args.protocol == "legacy-v1":
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
                "Refusing to overwrite existing legacy v1 SVM classification artifacts without --overwrite."
            )

        outputs = run_svm_experiments(
            project_root,
            primary_method=args.primary_method,
            comparators=args.comparators,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        outputs.grid_results.to_csv(
            output_dir / "svm_grid_results.csv", index=False, lineterminator="\n"
        )
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
            f"Stage 07 legacy v1 SVM evaluation complete | primary={args.primary_method} | "
            f"selected={sel['feature_set']} ({sel['kernel']}, C={sel['C']}, gamma={sel['gamma']}, cw={sel['class_weight']}) | "
            f"val_macro_f1={sel['val_macro_f1']:.4f} | val_accuracy={sel['val_accuracy']:.4f}"
        )
        return 0

    methods = list(args.methods) if args.methods is not None else list(COMPARISON_2X3_METHODS)
    if args.protocol == "2x3-strict":
        output_dir = project_root / CLASSIFICATION_2X3_STRICT_DIR_REL
        models_dir = project_root / MODELS_CV_2X3_STRICT_DIR_REL
        expected_model_ids = [
            "svm_shared_A",
            *(f"svm_{m}_B" for m in methods),
            *(f"svm_{m}_C" for m in methods),
            f"svm_{args.primary_method}_A_seg",
            f"svm_{args.primary_method}_C_seg",
        ]
        expected_files = [
            output_dir / "svm_2x3_strict_grid_results.csv",
            output_dir / "svm_2x3_strict_selected_summary.csv",
            output_dir / "svm_2x3_strict_per_category_val.csv",
            output_dir / "svm_2x3_strict_seg_vs_cls.csv",
            output_dir / "svm_2x3_strict_val_predictions.csv",
            output_dir / "svm_2x3_strict_val_bootstrap_models.csv",
            output_dir / "svm_2x3_strict_val_paired_contrasts.csv",
            output_dir / "svm_2x3_strict_vs_fixed_upstream.csv",
            output_dir / "fold_upstream_audit.json",
            output_dir / "classification_2x3_strict_manifest.json",
            *(models_dir / f"{model_id}.json" for model_id in expected_model_ids),
        ]
        if any(path.exists() for path in expected_files) and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing 2x3-strict SVM artifacts under '{output_dir}' without --overwrite."
            )

        strict_outputs = run_svm_2x3_strict_experiments(
            project_root,
            methods=methods,
            primary_method=args.primary_method,
            data_root_override=args.data_root,
            workers=args.workers,
            use_cache=not args.no_cache,
        )
        save_strict_2x3_outputs(strict_outputs, project_root)
        sel_rows = strict_outputs.selected_summary.loc[
            strict_outputs.selected_summary["model_variant"].eq("cv_selected")
        ]
        top_row = sel_rows.sort_values(["cv_macro_f1", "val_macro_f1"], ascending=False).iloc[0]
        print(
            f"Stage 07 2x3-strict evaluation complete | methods={','.join(methods)} | "
            f"top_by_cv={top_row['model_id']} (cv_macro_f1={top_row['cv_macro_f1']:.4f}, "
            f"val_macro_f1={top_row['val_macro_f1']:.4f} "
            f"[95% CI {top_row['val_macro_f1_ci95_low']:.4f}-{top_row['val_macro_f1_ci95_high']:.4f}], "
            f"val_accuracy={top_row['val_accuracy']:.4f})"
        )
        return 0

    output_dir = project_root / CLASSIFICATION_2X3_DIR_REL
    models_dir = project_root / MODELS_CV_2X3_DIR_REL

    expected_model_ids = [
        "svm_shared_A",
        *(f"svm_{m}_B" for m in methods),
        *(f"svm_{m}_C" for m in methods),
        f"svm_{args.primary_method}_A_seg",
        f"svm_{args.primary_method}_C_seg",
    ]
    expected_files = [
        output_dir / "svm_2x3_grid_results.csv",
        output_dir / "svm_2x3_selected_summary.csv",
        output_dir / "svm_2x3_per_category_val.csv",
        output_dir / "svm_2x3_segmentation_vs_classification.csv",
        output_dir / "svm_2x3_val_predictions.csv",
        output_dir / "classification_2x3_manifest.json",
        *(models_dir / f"{model_id}.json" for model_id in expected_model_ids),
    ]
    if any(path.exists() for path in expected_files) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing 2x3 SVM artifacts under '{output_dir}' without --overwrite."
        )

    outputs = run_svm_2x3_experiments(
        project_root,
        methods=methods,
        primary_method=args.primary_method,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    outputs.grid_results.to_csv(
        output_dir / "svm_2x3_grid_results.csv", index=False, lineterminator="\n"
    )
    outputs.selected_summary.to_csv(
        output_dir / "svm_2x3_selected_summary.csv", index=False, lineterminator="\n"
    )
    outputs.per_category_val.to_csv(
        output_dir / "svm_2x3_per_category_val.csv", index=False, lineterminator="\n"
    )
    outputs.segmentation_vs_classification.to_csv(
        output_dir / "svm_2x3_segmentation_vs_classification.csv",
        index=False,
        lineterminator="\n",
    )
    outputs.val_predictions.to_csv(
        output_dir / "svm_2x3_val_predictions.csv", index=False, lineterminator="\n"
    )

    with (output_dir / "classification_2x3_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(outputs.manifest, file, indent=2, allow_nan=False)
        file.write("\n")

    for model_id, payload in outputs.model_artifacts.items():
        model_path = models_dir / f"{model_id}.json"
        with model_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")

    best_cv = outputs.manifest["top_models_by_cv_selection"]["best_overall_by_cv"]
    print(
        f"Stage 07 2x3 CV-selected evaluation complete | methods={','.join(methods)} | "
        f"top_by_cv={best_cv['model_id']} (cv_macro_f1={best_cv['cv_macro_f1']:.4f}, "
        f"val_macro_f1={best_cv['val_macro_f1']:.4f}, val_accuracy={best_cv['val_accuracy']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
