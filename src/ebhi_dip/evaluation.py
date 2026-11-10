"""Stage 08 one-time held-out test evaluation for segmentation and six-class SVM classification.

Stage 08 contract:
- Evaluates the held-out ``test`` cohort (319 images, 53 groups across 6 categories) strictly once
  after locking all segmentation and classification decisions on ``train`` and ``val``.
- Never fits, tunes, rescales, or selects any parameter on ``val`` or ``test``.
- Segmentation uses only frozen ``configs/otsu_raw.json`` and ``models/kmeans_lab_k*_mapped.json``.
- Classification uses only frozen ``models/svm_*.json`` artifacts fitted on ``train`` (1,535 images).
- Reports group-blocked bootstrap 95% confidence intervals over the 53 ``test`` groups.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import pandas as pd
from PIL import __version__ as pillow_version
import sklearn

from ebhi_dip.analysis import (
    TIE_TOLERANCE,
    load_validation_results,
    method_label,
    summarize_validation_methods,
)
from ebhi_dip.classification import (
    PRIMARY_FEATURE_SETS,
    SENSITIVITY_FEATURE_SETS,
    classification_metrics,
    feature_columns_for_set,
    load_classification_feature_frame,
    load_svm_artifact,
    per_category_classification_table,
    predict_from_svm_artifact,
)
from ebhi_dip.dataset import (
    EXPECTED_CATEGORIES,
    REFERENCE_THRESHOLD,
    load_reference_mask,
    load_rgb,
    load_split_manifest,
)
from ebhi_dip.features import (
    DEFAULT_FEATURE_METHODS,
    FEATURE_NAMES,
    METADATA_COLUMNS,
    PRIMARY_FEATURE_METHOD,
    SEGMENTED_FEATURE_NAMES,
    validate_feature_table,
)
from ebhi_dip.metrics import (
    binary_segmentation_metrics,
    category_macro_scores,
    category_score_table,
)
from ebhi_dip.paths import resolve_manifest_path, verify_frozen_split
from ebhi_dip.provenance import sha256_bytes
from ebhi_dip.segmentation import (
    STAGES,
    SUPPORTED_METHODS,
    load_kmeans_model,
    load_otsu_config,
    morphology_variants,
    nearest_clusters,
    otsu_threshold_mask,
    rgb_to_lab,
)

SEGMENTATION_SELECTION_REL = Path("configs/segmentation_selection.json")
TEST_RESULTS_DIR_REL = Path("results/test")
DEFAULT_BOOTSTRAP_RESAMPLES: int = 2000
DEFAULT_BOOTSTRAP_SEED: int = 42


@dataclass(frozen=True)
class TestSegmentationOutputs:
    """Container for held-out test segmentation tables and provenance."""

    per_image: pd.DataFrame
    macro_summary: pd.DataFrame
    per_category: pd.DataFrame
    paired_contrasts: pd.DataFrame


@dataclass(frozen=True)
class TestClassificationOutputs:
    """Container for held-out test SVM classification tables and predictions."""

    summary: pd.DataFrame
    per_category: pd.DataFrame
    paired_contrasts: pd.DataFrame
    predictions: pd.DataFrame


@dataclass(frozen=True)
class FinalTestEvaluationOutputs:
    """Complete Stage 08 held-out test evaluation outputs."""

    segmentation: TestSegmentationOutputs
    classification: TestClassificationOutputs
    selection_config: dict[str, object]
    manifest: dict[str, object]
    split_sha256: str


def build_segmentation_selection_config(project_root: str | Path) -> dict[str, object]:
    """Construct the canonical pre-test ``configs/segmentation_selection.json`` payload."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    split_df = load_split_manifest(root, verify=False)

    val_results = load_validation_results(root)
    val_summary = summarize_validation_methods(val_results)

    cls_manifest_path = root / "results/classification/classification_manifest.json"
    if not cls_manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage 07 classification manifest: {cls_manifest_path}")
    cls_manifest = json.loads(cls_manifest_path.read_text(encoding="utf-8"))

    comparators = list(DEFAULT_FEATURE_METHODS)
    method_entries: dict[str, dict[str, float]] = {}
    for method_id in (*comparators, "all_foreground"):
        row = val_summary.loc[method_id]
        method_entries[method_id] = {
            "mean_category_dice": float(row["mean_category_dice"]),
            "mean_category_iou": float(row["mean_category_iou"]),
            "delta_dice_vs_control": float(row["delta_dice_vs_control"]),
            "delta_iou_vs_control": float(row["delta_iou_vs_control"]),
        }

    train_sub = split_df.loc[split_df["split"].eq("train")]
    val_sub = split_df.loc[split_df["split"].eq("val")]
    test_sub = split_df.loc[split_df["split"].eq("test")]

    return {
        "schema_version": "segmentation-selection-v1",
        "selection_split": "val",
        "split_sha256": split_sha,
        "cohort_counts": {
            "train_images": int(len(train_sub)),
            "train_groups": int(train_sub["group_id"].nunique()),
            "val_images": int(len(val_sub)),
            "val_groups": int(val_sub["group_id"].nunique()),
            "test_images_held_out": int(len(test_sub)),
            "test_groups_held_out": int(test_sub["group_id"].nunique()),
        },
        "selection_rule": (
            "mean_category_dice (4 decimals) -> mean_category_iou (4 decimals) "
            "-> fewer morphology stages -> fewer clusters"
        ),
        "primary_method": PRIMARY_FEATURE_METHOD,
        "comparators": comparators,
        "negative_control": "all_foreground",
        "validation_summary": method_entries,
        "downstream_classification_lock": {
            "primary_track": {
                "segmentation_method": PRIMARY_FEATURE_METHOD,
                "feature_sets": list(PRIMARY_FEATURE_SETS),
                "selected_model_id": cls_manifest["selected_primary_model"]["model_id"],
                "selected_model_file": "models/svm_k4_M0_selected.json",
                "selected_feature_set": cls_manifest["selected_primary_model"]["feature_set"],
                "kernel": cls_manifest["selected_primary_model"]["kernel"],
                "C": float(cls_manifest["selected_primary_model"]["C"]),
                "gamma": str(cls_manifest["selected_primary_model"]["gamma"]),
                "class_weight": cls_manifest["selected_primary_model"]["class_weight"],
                "val_macro_f1": float(cls_manifest["selected_primary_model"]["val_macro_f1"]),
                "val_balanced_accuracy": float(
                    cls_manifest["selected_primary_model"]["val_balanced_accuracy"]
                ),
                "val_accuracy": float(cls_manifest["selected_primary_model"]["val_accuracy"]),
            },
            "pre_registered_sensitivity_tracks": {
                "segmented_appearance": {
                    "segmentation_method": PRIMARY_FEATURE_METHOD,
                    "feature_sets": list(SENSITIVITY_FEATURE_SETS),
                    "model_files": [
                        "models/svm_k4_M0_A_seg.json",
                        "models/svm_k4_M0_C_seg.json",
                    ],
                },
                "segmentation_comparators": {
                    "methods": comparators,
                    "feature_sets": ["B", "C"],
                    "spec": "primary_tuned_spec (rbf, C=10.0, gamma=0.01, class_weight=none)",
                    "best_validation_comparator_method": "otsu_M2",
                    "best_validation_comparator_model_file": "models/svm_otsu_M2_C.json",
                    "model_files": [
                        "models/svm_otsu_M0_B.json",
                        "models/svm_otsu_M0_C.json",
                        "models/svm_otsu_M2_B.json",
                        "models/svm_otsu_M2_C.json",
                        "models/svm_k4_M1_B.json",
                        "models/svm_k4_M1_C.json",
                    ],
                },
            },
        },
        "input_artifact_sha256": {
            "split_manifest": sha256_bytes(root / "data/metadata/split_manifest.csv"),
            "otsu_config": sha256_bytes(root / "configs/otsu_raw.json"),
            "kmeans_k2_model": sha256_bytes(root / "models/kmeans_lab_k2_mapped.json"),
            "kmeans_k3_model": sha256_bytes(root / "models/kmeans_lab_k3_mapped.json"),
            "kmeans_k4_model": sha256_bytes(root / "models/kmeans_lab_k4_mapped.json"),
            "otsu_morph_val": sha256_bytes(root / "results/otsu_morph_val.csv"),
            "kmeans_morph_val": sha256_bytes(root / "results/kmeans_morph_val.csv"),
            "kmeans_morph_val_run": sha256_bytes(root / "results/kmeans_morph_val_run.json"),
            "classification_manifest": sha256_bytes(cls_manifest_path),
        },
    }


def load_segmentation_selection_config(project_root: str | Path) -> dict[str, object]:
    """Load and validate ``configs/segmentation_selection.json`` against frozen artifacts."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    config_path = root / SEGMENTATION_SELECTION_REL
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing segmentation selection config: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "segmentation-selection-v1":
        raise ValueError(
            f"Unsupported segmentation selection schema: {payload.get('schema_version')!r}."
        )
    if payload.get("selection_split") != "val" or payload.get("split_sha256") != split_sha:
        raise ValueError("Segmentation selection config split or SHA-256 mismatch.")
    if payload.get("primary_method") != PRIMARY_FEATURE_METHOD:
        raise ValueError(
            f"Unexpected primary_method in segmentation_selection.json: {payload.get('primary_method')!r}."
        )
    return payload


def load_test_feature_frame(
    project_root: str | Path,
    method: str = PRIMARY_FEATURE_METHOD,
    segmented: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load and validate strictly the held-out ``test`` (319 rows, 53 groups) feature table."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    split_df = load_split_manifest(root, verify=False)

    feature_dir = root / "results/features"
    if segmented:
        csv_path = feature_dir / f"features_{method}_segmented.csv"
        expected_cols = SEGMENTED_FEATURE_NAMES
    else:
        csv_path = feature_dir / f"features_{method}.csv"
        expected_cols = FEATURE_NAMES

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing canonical feature table: {csv_path}")

    table = pd.read_csv(csv_path)
    validate_feature_table(table, feature_names=expected_cols, require_method=True)
    if not table["method"].eq(method).all():
        raise ValueError(f"Feature table {csv_path.name} contains method values other than {method!r}.")

    merged_check = table[list(METADATA_COLUMNS)].merge(
        split_df[list(METADATA_COLUMNS)],
        on="sample_id",
        suffixes=("_feat", "_split"),
        how="outer",
        validate="one_to_one",
    )
    for col in ("category", "group_id", "split"):
        if not merged_check[f"{col}_feat"].eq(merged_check[f"{col}_split"]).all():
            raise ValueError(f"Metadata mismatch on column '{col}' in {csv_path.name}.")

    train_ids = set(table.loc[table["split"].eq("train"), "sample_id"])
    val_ids = set(table.loc[table["split"].eq("val"), "sample_id"])
    train_groups = set(table.loc[table["split"].eq("train"), "group_id"])
    val_groups = set(table.loc[table["split"].eq("val"), "group_id"])

    test_df = (
        table.loc[table["split"].eq("test")]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(test_df) != 319 or int(test_df["group_id"].nunique()) != 53:
        raise ValueError(
            f"Unexpected test cohort counts: rows={len(test_df)}, groups={test_df['group_id'].nunique()}."
        )
    if not set(test_df["sample_id"]).isdisjoint(train_ids | val_ids):
        raise ValueError("Held-out test sample_id set overlaps train or val.")
    if not set(test_df["group_id"]).isdisjoint(train_groups | val_groups):
        raise ValueError("Held-out test group_id set overlaps train or val.")

    return test_df, split_sha


def _group_bootstrap_indices(
    groups: Sequence[str] | np.ndarray,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[np.ndarray]:
    """Generate deterministic cluster-bootstrap row index arrays by resampling unique group_ids."""
    g_arr = np.asarray(groups, dtype=object)
    unique_groups = np.array(sorted(set(g_arr)), dtype=object)
    n_groups = len(unique_groups)
    group_to_rows: dict[str, np.ndarray] = {
        str(g): np.flatnonzero(g_arr == g) for g in unique_groups
    }
    rng = np.random.default_rng(seed)
    index_list: list[np.ndarray] = []
    for _ in range(int(n_resamples)):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        rows = np.concatenate([group_to_rows[str(g)] for g in sampled_groups])
        index_list.append(rows)
    return index_list


def _bootstrap_macro_segmentation_ci(
    per_image_df: pd.DataFrame,
    methods: Sequence[str],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Compute group-blocked 95% bootstrap CIs for equal-category macro Dice and IoU."""
    pivot_dice = per_image_df.pivot(index="sample_id", columns="method", values="dice").sort_index()
    pivot_iou = per_image_df.pivot(index="sample_id", columns="method", values="iou").sort_index()
    meta = (
        per_image_df[["sample_id", "category", "group_id"]]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
        .loc[pivot_dice.index]
    )
    cats = meta["category"].to_numpy(dtype=object)
    groups = meta["group_id"].to_numpy(dtype=object)
    boot_indices = _group_bootstrap_indices(groups, n_resamples=n_resamples, seed=seed)

    dice_mat = pivot_dice[list(methods)].to_numpy(dtype=np.float64)
    iou_mat = pivot_iou[list(methods)].to_numpy(dtype=np.float64)

    boot_dice = np.zeros((len(boot_indices), len(methods)), dtype=np.float64)
    boot_iou = np.zeros((len(boot_indices), len(methods)), dtype=np.float64)

    for b_idx, rows in enumerate(boot_indices):
        b_cats = cats[rows]
        b_dice = dice_mat[rows]
        b_iou = iou_mat[rows]
        cat_dice_means = []
        cat_iou_means = []
        for cat in EXPECTED_CATEGORIES:
            mask_c = b_cats == cat
            if np.any(mask_c):
                cat_dice_means.append(b_dice[mask_c].mean(axis=0))
                cat_iou_means.append(b_iou[mask_c].mean(axis=0))
        boot_dice[b_idx] = np.mean(cat_dice_means, axis=0)
        boot_iou[b_idx] = np.mean(cat_iou_means, axis=0)

    ci_map: dict[str, dict[str, float]] = {}
    for m_idx, m_name in enumerate(methods):
        ci_map[m_name] = {
            "dice_ci95_low": float(np.percentile(boot_dice[:, m_idx], 2.5)),
            "dice_ci95_high": float(np.percentile(boot_dice[:, m_idx], 97.5)),
            "iou_ci95_low": float(np.percentile(boot_iou[:, m_idx], 2.5)),
            "iou_ci95_high": float(np.percentile(boot_iou[:, m_idx], 97.5)),
        }
    return ci_map


def _paired_segmentation_contrasts(
    per_image_df: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Compute paired macro Dice/IoU differences, group-bootstrap 95% CIs, and per-image win/loss/ties."""
    pivot_dice = per_image_df.pivot(index="sample_id", columns="method", values="dice").sort_index()
    pivot_iou = per_image_df.pivot(index="sample_id", columns="method", values="iou").sort_index()
    meta = (
        per_image_df[["sample_id", "category", "group_id"]]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
        .loc[pivot_dice.index]
    )
    cats = meta["category"].to_numpy(dtype=object)
    groups = meta["group_id"].to_numpy(dtype=object)
    boot_indices = _group_bootstrap_indices(groups, n_resamples=n_resamples, seed=seed)

    macro = category_macro_scores(per_image_df, expected_categories=EXPECTED_CATEGORIES)
    rows: list[dict[str, object]] = []

    for method_a, method_b in pairs:
        d_a = pivot_dice[method_a].to_numpy(dtype=np.float64)
        d_b = pivot_dice[method_b].to_numpy(dtype=np.float64)
        i_a = pivot_iou[method_a].to_numpy(dtype=np.float64)
        i_b = pivot_iou[method_b].to_numpy(dtype=np.float64)

        diff_d_img = d_a - d_b
        diff_i_img = i_a - i_b

        a_wins = int(np.sum(diff_d_img > TIE_TOLERANCE))
        b_wins = int(np.sum(diff_d_img < -TIE_TOLERANCE))
        ties = int(len(diff_d_img) - a_wins - b_wins)

        boot_macro_diff_d = np.zeros(len(boot_indices), dtype=np.float64)
        boot_macro_diff_i = np.zeros(len(boot_indices), dtype=np.float64)
        for b_idx, r_idx in enumerate(boot_indices):
            b_cats = cats[r_idx]
            cat_d = [
                float(diff_d_img[r_idx][b_cats == c].mean())
                for c in EXPECTED_CATEGORIES
                if np.any(b_cats == c)
            ]
            cat_i = [
                float(diff_i_img[r_idx][b_cats == c].mean())
                for c in EXPECTED_CATEGORIES
                if np.any(b_cats == c)
            ]
            boot_macro_diff_d[b_idx] = float(np.mean(cat_d))
            boot_macro_diff_i[b_idx] = float(np.mean(cat_i))

        macro_diff_dice = float(
            macro.loc[method_a, "mean_category_dice"] - macro.loc[method_b, "mean_category_dice"]
        )
        macro_diff_iou = float(
            macro.loc[method_a, "mean_category_iou"] - macro.loc[method_b, "mean_category_iou"]
        )
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "contrast_label": f"{method_label(method_a)} vs {method_label(method_b)}",
                "macro_dice_a": float(macro.loc[method_a, "mean_category_dice"]),
                "macro_dice_b": float(macro.loc[method_b, "mean_category_dice"]),
                "delta_macro_dice_a_minus_b": macro_diff_dice,
                "delta_macro_dice_ci95_low": float(np.percentile(boot_macro_diff_d, 2.5)),
                "delta_macro_dice_ci95_high": float(np.percentile(boot_macro_diff_d, 97.5)),
                "macro_iou_a": float(macro.loc[method_a, "mean_category_iou"]),
                "macro_iou_b": float(macro.loc[method_b, "mean_category_iou"]),
                "delta_macro_iou_a_minus_b": macro_diff_iou,
                "delta_macro_iou_ci95_low": float(np.percentile(boot_macro_diff_i, 2.5)),
                "delta_macro_iou_ci95_high": float(np.percentile(boot_macro_diff_i, 97.5)),
                "mean_image_delta_dice": float(diff_d_img.mean()),
                "median_image_delta_dice": float(np.median(diff_d_img)),
                "a_higher_count": a_wins,
                "b_higher_count": b_wins,
                "tie_count": ties,
            }
        )
    return pd.DataFrame(rows)


def evaluate_test_segmentation(
    project_root: str | Path,
    data_root_override: str | Path | None = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> TestSegmentationOutputs:
    """Evaluate all 13 frozen segmentation methods on the 319 held-out ``test`` images."""
    root = Path(project_root).expanduser().resolve()
    verify_frozen_split(root)
    split_df = load_split_manifest(root, verify=False)
    test_cohort = (
        split_df.loc[split_df["split"].eq("test")]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(test_cohort) != 319 or int(test_cohort["group_id"].nunique()) != 53:
        raise ValueError("Unexpected held-out test cohort size in split_manifest.csv.")

    otsu_config = load_otsu_config(root)
    polarity = str(otsu_config["foreground_polarity"])
    kmeans_models = {k: load_kmeans_model(root, k=k, verify_provenance=True) for k in (2, 3, 4)}

    records: list[dict[str, object]] = []
    for _, rec in test_cohort.iterrows():
        img = load_rgb(
            resolve_manifest_path(root, rec["image_path"], data_root_override=data_root_override)
        )
        ref = load_reference_mask(
            resolve_manifest_path(root, rec["mask_path"], data_root_override=data_root_override),
            threshold=REFERENCE_THRESHOLD,
        )
        thr, otsu_raw = otsu_threshold_mask(img, foreground_polarity=polarity)
        otsu_vars = morphology_variants(otsu_raw)

        lab = rgb_to_lab(img)
        flat_lab = lab.reshape(-1, 3)
        masks_by_method: dict[str, np.ndarray] = {
            f"otsu_{st}": otsu_vars[st] for st in STAGES
        }
        for k, model in kmeans_models.items():
            labels = nearest_clusters(flat_lab, model["cluster_centers"]).reshape(img.shape[:2])
            km_raw = np.isin(labels, model["foreground_clusters"])
            km_vars = morphology_variants(km_raw)
            for st in STAGES:
                masks_by_method[f"k{k}_{st}"] = km_vars[st]
        masks_by_method["all_foreground"] = np.ones_like(ref, dtype=bool)

        for method_id in SUPPORTED_METHODS:
            m_stats = binary_segmentation_metrics(masks_by_method[method_id], ref)
            records.append(
                {
                    "sample_id": rec["sample_id"],
                    "category": rec["category"],
                    "group_id": rec["group_id"],
                    "split": "test",
                    "method": method_id,
                    "dice": float(m_stats["dice"]),
                    "iou": float(m_stats["iou"]),
                    "precision": float(m_stats["precision"]),
                    "recall": float(m_stats["recall"]),
                    "predicted_foreground_fraction": float(
                        m_stats["predicted_foreground_fraction"]
                    ),
                    "reference_foreground_fraction": float(
                        m_stats["reference_foreground_fraction"]
                    ),
                    "tp": int(m_stats["tp"]),
                    "fp": int(m_stats["fp"]),
                    "fn": int(m_stats["fn"]),
                    "tn": int(m_stats["tn"]),
                    "otsu_threshold": float(thr),
                }
            )

    per_image_df = pd.DataFrame(records)

    # Verify exact mask_foreground_fraction parity with canonical Stage 06 test feature tables
    for feat_method in DEFAULT_FEATURE_METHODS:
        test_feat_df, _ = load_test_feature_frame(root, method=feat_method, segmented=False)
        seg_sub = (
            per_image_df.loc[per_image_df["method"].eq(feat_method)]
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        np.testing.assert_allclose(
            seg_sub["predicted_foreground_fraction"].to_numpy(dtype=np.float64),
            test_feat_df["mask_foreground_fraction"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )

    # Compare against validation summary
    val_results = load_validation_results(root)
    val_macro = summarize_validation_methods(val_results)
    val_per_cat = category_score_table(val_results, expected_categories=EXPECTED_CATEGORIES)

    test_macro_raw = category_macro_scores(per_image_df, expected_categories=EXPECTED_CATEGORIES)
    control_test_dice = float(test_macro_raw.loc["all_foreground", "mean_category_dice"])
    control_test_iou = float(test_macro_raw.loc["all_foreground", "mean_category_iou"])

    ci_map = _bootstrap_macro_segmentation_ci(
        per_image_df,
        methods=SUPPORTED_METHODS,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )

    macro_rows: list[dict[str, object]] = []
    for method_id in SUPPORTED_METHODS:
        m_sub = per_image_df.loc[per_image_df["method"].eq(method_id)]
        v_dice = float(val_macro.loc[method_id, "mean_category_dice"])
        v_iou = float(val_macro.loc[method_id, "mean_category_iou"])
        t_dice = float(test_macro_raw.loc[method_id, "mean_category_dice"])
        t_iou = float(test_macro_raw.loc[method_id, "mean_category_iou"])
        ci_info = ci_map[method_id]

        macro_rows.append(
            {
                "method": method_id,
                "label": method_label(method_id),
                "is_primary_selected": method_id == PRIMARY_FEATURE_METHOD,
                "is_key_comparator": method_id in (*DEFAULT_FEATURE_METHODS, "all_foreground"),
                "val_mean_category_dice": v_dice,
                "test_mean_category_dice": t_dice,
                "test_dice_ci95_low": ci_info["dice_ci95_low"],
                "test_dice_ci95_high": ci_info["dice_ci95_high"],
                "generalization_delta_dice": t_dice - v_dice,
                "test_delta_dice_vs_control": t_dice - control_test_dice,
                "val_mean_category_iou": v_iou,
                "test_mean_category_iou": t_iou,
                "test_iou_ci95_low": ci_info["iou_ci95_low"],
                "test_iou_ci95_high": ci_info["iou_ci95_high"],
                "generalization_delta_iou": t_iou - v_iou,
                "test_delta_iou_vs_control": t_iou - control_test_iou,
                "test_mean_image_dice": float(m_sub["dice"].mean()),
                "test_mean_image_iou": float(m_sub["iou"].mean()),
                "test_mean_precision": float(m_sub["precision"].dropna().mean()),
                "test_mean_recall": float(m_sub["recall"].dropna().mean()),
                "test_mean_pred_fg_fraction": float(
                    m_sub["predicted_foreground_fraction"].mean()
                ),
                "test_empty_mask_count": int(
                    (m_sub["predicted_foreground_fraction"] == 0.0).sum()
                ),
            }
        )
    macro_summary_df = pd.DataFrame(macro_rows).sort_values(
        ["test_mean_category_dice", "test_mean_category_iou"], ascending=False
    ).reset_index(drop=True)

    test_per_cat_raw = category_score_table(per_image_df, expected_categories=EXPECTED_CATEGORIES)
    per_cat_rows: list[dict[str, object]] = []
    for method_id in SUPPORTED_METHODS:
        for cat in EXPECTED_CATEGORIES:
            sub_mc = per_image_df.loc[
                per_image_df["method"].eq(method_id) & per_image_df["category"].eq(cat)
            ]
            v_d = float(val_per_cat.loc[(method_id, cat), "dice"])
            v_i = float(val_per_cat.loc[(method_id, cat), "iou"])
            t_d = float(test_per_cat_raw.loc[(method_id, cat), "dice"])
            t_i = float(test_per_cat_raw.loc[(method_id, cat), "iou"])
            per_cat_rows.append(
                {
                    "method": method_id,
                    "category": cat,
                    "test_image_count": int(len(sub_mc)),
                    "test_group_count": int(sub_mc["group_id"].nunique()),
                    "val_dice": v_d,
                    "test_dice": t_d,
                    "delta_dice_test_minus_val": t_d - v_d,
                    "val_iou": v_i,
                    "test_iou": t_i,
                    "delta_iou_test_minus_val": t_i - v_i,
                    "test_mean_precision": float(sub_mc["precision"].dropna().mean()),
                    "test_mean_recall": float(sub_mc["recall"].dropna().mean()),
                    "test_mean_pred_fg_fraction": float(
                        sub_mc["predicted_foreground_fraction"].mean()
                    ),
                    "test_mean_ref_fg_fraction": float(
                        sub_mc["reference_foreground_fraction"].mean()
                    ),
                }
            )
    per_cat_df = pd.DataFrame(per_cat_rows)

    contrast_pairs = (
        ("k4_M0", "otsu_M0"),
        ("k4_M0", "otsu_M2"),
        ("k4_M1", "otsu_M2"),
        ("k4_M0", "k4_M1"),
        ("k4_M0", "all_foreground"),
        ("otsu_M2", "otsu_M0"),
    )
    paired_contrasts_df = _paired_segmentation_contrasts(
        per_image_df,
        pairs=contrast_pairs,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )

    return TestSegmentationOutputs(
        per_image=per_image_df,
        macro_summary=macro_summary_df,
        per_category=per_cat_df,
        paired_contrasts=paired_contrasts_df,
    )


def _fast_classification_summary(
    y_true_int: np.ndarray,
    y_pred_int: np.ndarray,
    n_classes: int = 6,
) -> tuple[float, float, float]:
    """Compute (macro_f1, balanced_accuracy, accuracy) from integer class indices in [0, n_classes)."""
    cm = np.bincount(
        n_classes * y_true_int + y_pred_int,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)
    denom = support + pred_count
    f1_per_class = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0.0)
    macro_f1 = float(f1_per_class.mean())
    present = support > 0.0
    recalls = np.divide(tp[present], support[present])
    balanced_acc = float(recalls.mean()) if recalls.size > 0 else 0.0
    acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0
    return macro_f1, balanced_acc, acc


def evaluate_test_classification(
    project_root: str | Path,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> TestClassificationOutputs:
    """Evaluate frozen Stage 07 SVM JSON artifacts on the 319 held-out ``test`` images."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    models_dir = root / "models"
    cls_dir = root / "results/classification"

    # Pre-load validation and test feature tables for all 4 methods + segmented k4_M0
    val_tables: dict[tuple[str, bool], pd.DataFrame] = {}
    test_tables: dict[tuple[str, bool], pd.DataFrame] = {}
    for m_name in DEFAULT_FEATURE_METHODS:
        _, v_df, _ = load_classification_feature_frame(root, method=m_name, segmented=False)
        t_df, _ = load_test_feature_frame(root, method=m_name, segmented=False)
        val_tables[(m_name, False)] = v_df
        test_tables[(m_name, False)] = t_df
    _, v_seg_df, _ = load_classification_feature_frame(root, method=PRIMARY_FEATURE_METHOD, segmented=True)
    t_seg_df, _ = load_test_feature_frame(root, method=PRIMARY_FEATURE_METHOD, segmented=True)
    val_tables[(PRIMARY_FEATURE_METHOD, True)] = v_seg_df
    test_tables[(PRIMARY_FEATURE_METHOD, True)] = t_seg_df

    # Frozen models to evaluate on test (never fitting or tuning)
    model_specs = (
        ("primary_k4_M0", "svm_k4_M0_A", "k4_M0", "A", False),
        ("primary_k4_M0", "svm_k4_M0_B", "k4_M0", "B", False),
        ("primary_k4_M0", "svm_k4_M0_C", "k4_M0", "C", False),
        ("primary_selected", "svm_k4_M0_selected", "k4_M0", "C", False),
        ("sensitivity_A_seg_C_seg", "svm_k4_M0_A_seg", "k4_M0", "A_seg", True),
        ("sensitivity_A_seg_C_seg", "svm_k4_M0_C_seg", "k4_M0", "C_seg", True),
        ("comparator_tuned_spec", "svm_otsu_M0_B", "otsu_M0", "B", False),
        ("comparator_tuned_spec", "svm_otsu_M0_C", "otsu_M0", "C", False),
        ("comparator_tuned_spec", "svm_otsu_M2_B", "otsu_M2", "B", False),
        ("comparator_tuned_spec", "svm_otsu_M2_C", "otsu_M2", "C", False),
        ("comparator_tuned_spec", "svm_k4_M1_B", "k4_M1", "B", False),
        ("comparator_tuned_spec", "svm_k4_M1_C", "k4_M1", "C", False),
    )

    base_test_df = test_tables[(PRIMARY_FEATURE_METHOD, False)]
    y_true = base_test_df["category"].to_numpy(dtype=object)
    cat_to_int = {cat: idx for idx, cat in enumerate(EXPECTED_CATEGORIES)}
    y_true_int = np.array([cat_to_int[str(c)] for c in y_true], dtype=np.int64)
    groups = base_test_df["group_id"].to_numpy(dtype=object)
    boot_indices = _group_bootstrap_indices(groups, n_resamples=bootstrap_resamples, seed=seed)

    summary_rows: list[dict[str, object]] = []
    per_cat_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    preds_by_model: dict[str, np.ndarray] = {}
    preds_int_by_model: dict[str, np.ndarray] = {}

    for track, model_id, method_id, set_name, is_segmented in model_specs:
        model_path = models_dir / f"{model_id}.json"
        artifact = load_svm_artifact(model_path)

        # Strict provenance guards: model must be fitted on train (1,535) of the frozen split
        if artifact.get("split_sha256") != split_sha:
            raise ValueError(f"Split SHA-256 mismatch in {model_path.name}.")
        if artifact.get("segmentation_method") != method_id or artifact.get("feature_set") != set_name:
            raise ValueError(f"Metadata mismatch in {model_path.name}.")
        if int(artifact["scaler"].get("n_samples_seen", 0)) != 1535:
            raise ValueError(
                f"Model {model_path.name} scaler was not fitted on the 1,535 training samples."
            )

        t_df = test_tables[(method_id, is_segmented)]
        cols = feature_columns_for_set(set_name)
        if list(artifact["feature_names"]) != list(cols):
            raise ValueError(f"Feature column ordering mismatch in {model_path.name}.")

        X_test_raw = t_df[list(cols)].to_numpy(dtype=np.float64)
        pred_labels, _, votes = predict_from_svm_artifact(X_test_raw, artifact)
        pred_int = np.array([cat_to_int[str(c)] for c in pred_labels], dtype=np.int64)
        preds_by_model[model_id] = pred_labels
        preds_int_by_model[model_id] = pred_int

        test_m = classification_metrics(y_true, pred_labels)
        val_m = artifact["validation_metrics"]
        cv_m = artifact.get("train_cv_metrics", {})

        # Group-blocked bootstrap 95% CIs on test
        boot_f1 = np.zeros(len(boot_indices), dtype=np.float64)
        boot_bacc = np.zeros(len(boot_indices), dtype=np.float64)
        boot_acc = np.zeros(len(boot_indices), dtype=np.float64)
        for b_idx, r_idx in enumerate(boot_indices):
            bf1, bbacc, bacc = _fast_classification_summary(
                y_true_int[r_idx], pred_int[r_idx], n_classes=len(EXPECTED_CATEGORIES)
            )
            boot_f1[b_idx] = bf1
            boot_bacc[b_idx] = bbacc
            boot_acc[b_idx] = bacc

        summary_rows.append(
            {
                "track": track,
                "model_id": model_id,
                "method": method_id,
                "feature_set": set_name,
                "n_features": len(cols),
                "kernel": str(artifact["kernel"]),
                "class_weight": str(artifact["class_weight"]),
                "C": float(artifact["C"]),
                "gamma": str(artifact["gamma_param"]),
                "cv_macro_f1": float(cv_m.get("macro_f1", np.nan)),
                "val_macro_f1": float(val_m["macro_f1"]),
                "test_macro_f1": float(test_m["macro_f1"]),
                "test_macro_f1_ci95_low": float(np.percentile(boot_f1, 2.5)),
                "test_macro_f1_ci95_high": float(np.percentile(boot_f1, 97.5)),
                "generalization_delta_macro_f1": float(test_m["macro_f1"] - float(val_m["macro_f1"])),
                "val_balanced_accuracy": float(val_m["balanced_accuracy"]),
                "test_balanced_accuracy": float(test_m["balanced_accuracy"]),
                "test_balanced_accuracy_ci95_low": float(np.percentile(boot_bacc, 2.5)),
                "test_balanced_accuracy_ci95_high": float(np.percentile(boot_bacc, 97.5)),
                "generalization_delta_balanced_accuracy": float(
                    test_m["balanced_accuracy"] - float(val_m["balanced_accuracy"])
                ),
                "val_accuracy": float(val_m["accuracy"]),
                "test_accuracy": float(test_m["accuracy"]),
                "test_accuracy_ci95_low": float(np.percentile(boot_acc, 2.5)),
                "test_accuracy_ci95_high": float(np.percentile(boot_acc, 97.5)),
                "generalization_delta_accuracy": float(test_m["accuracy"] - float(val_m["accuracy"])),
                "val_weighted_f1": float(val_m["weighted_f1"]),
                "test_weighted_f1": float(test_m["weighted_f1"]),
                "test_macro_precision": float(test_m["macro_precision"]),
                "n_support_total": int(sum(artifact["n_support"])),
            }
        )

        cat_df = per_category_classification_table(y_true, pred_labels)
        cat_df.insert(0, "feature_set", set_name)
        cat_df.insert(0, "method", method_id)
        cat_df.insert(0, "model_id", model_id)
        cat_df.insert(0, "track", track)

        # Attach validation recall and F1 from the same frozen artifact on val
        v_df = val_tables[(method_id, is_segmented)]
        X_val_raw = v_df[list(cols)].to_numpy(dtype=np.float64)
        y_val = v_df["category"].to_numpy(dtype=object)
        val_pred_labels, _, _ = predict_from_svm_artifact(X_val_raw, artifact)
        val_cat_df = per_category_classification_table(y_val, val_pred_labels)
        val_f1_map = val_cat_df.set_index("category")["f1"].to_dict()
        val_rec_map = val_cat_df.set_index("category")["recall"].to_dict()
        cat_df["val_recall"] = cat_df["category"].map(val_rec_map)
        cat_df["val_f1"] = cat_df["category"].map(val_f1_map)
        cat_df["delta_f1_test_minus_val"] = cat_df["f1"] - cat_df["val_f1"]
        per_cat_frames.append(cat_df)

        sorted_votes = np.sort(votes, axis=1)
        vote_margin = sorted_votes[:, -1] - sorted_votes[:, -2]
        max_votes = sorted_votes[:, -1]

        pred_df = t_df[list(METADATA_COLUMNS)].copy()
        pred_df["track"] = track
        pred_df["model_id"] = model_id
        pred_df["method"] = method_id
        pred_df["feature_set"] = set_name
        pred_df["predicted_category"] = pred_labels
        pred_df["correct"] = pred_df["category"].to_numpy() == pred_labels
        pred_df["winning_votes"] = max_votes.astype(int)
        pred_df["vote_margin"] = vote_margin.astype(int)
        pred_frames.append(pred_df)

    # Paired bootstrap contrasts on held-out test
    cls_contrast_specs = (
        ("svm_k4_M0_C", "svm_k4_M0_A", "k4_M0: Set C vs Set A (efek tambahan morfologi k4_M0)"),
        ("svm_k4_M0_C", "svm_k4_M0_B", "k4_M0: Set C vs Set B (efek tambahan appearance pada k4_M0)"),
        ("svm_k4_M0_C", "svm_k4_M0_C_seg", "k4_M0: Set C vs Set C_seg (whole-image vs masked appearance)"),
        ("svm_otsu_M2_C", "svm_k4_M0_C", "Set C: otsu_M2 vs k4_M0 (sensitivitas komparator utama)"),
        ("svm_otsu_M2_B", "svm_k4_M0_B", "Set B: otsu_M2 vs k4_M0 (morfologi murni otsu_M2 vs k4_M0)"),
        ("svm_otsu_M2_C", "svm_otsu_M0_C", "Set C: otsu_M2 vs otsu_M0 (efek pembersihan morfologi M2)"),
    )
    summary_indexed = pd.DataFrame(summary_rows).set_index("model_id")
    contrast_rows: list[dict[str, object]] = []
    for mod_a, mod_b, desc in cls_contrast_specs:
        pa = preds_by_model[mod_a]
        pb = preds_by_model[mod_b]
        pa_int = preds_int_by_model[mod_a]
        pb_int = preds_int_by_model[mod_b]
        boot_diff_f1 = np.zeros(len(boot_indices), dtype=np.float64)
        boot_diff_acc = np.zeros(len(boot_indices), dtype=np.float64)
        for b_idx, r_idx in enumerate(boot_indices):
            f1_a_b, _, acc_a_b = _fast_classification_summary(
                y_true_int[r_idx], pa_int[r_idx], n_classes=len(EXPECTED_CATEGORIES)
            )
            f1_b_b, _, acc_b_b = _fast_classification_summary(
                y_true_int[r_idx], pb_int[r_idx], n_classes=len(EXPECTED_CATEGORIES)
            )
            boot_diff_f1[b_idx] = f1_a_b - f1_b_b
            boot_diff_acc[b_idx] = acc_a_b - acc_b_b

        f1_a = float(summary_indexed.loc[mod_a, "test_macro_f1"])
        f1_b = float(summary_indexed.loc[mod_b, "test_macro_f1"])
        acc_a = float(summary_indexed.loc[mod_a, "test_accuracy"])
        acc_b = float(summary_indexed.loc[mod_b, "test_accuracy"])
        a_only_correct = int(np.sum((pa == y_true) & (pb != y_true)))
        b_only_correct = int(np.sum((pa != y_true) & (pb == y_true)))
        both_correct = int(np.sum((pa == y_true) & (pb == y_true)))
        both_wrong = int(np.sum((pa != y_true) & (pb != y_true)))

        contrast_rows.append(
            {
                "model_a": mod_a,
                "model_b": mod_b,
                "description": desc,
                "test_macro_f1_a": f1_a,
                "test_macro_f1_b": f1_b,
                "delta_macro_f1_a_minus_b": f1_a - f1_b,
                "delta_macro_f1_ci95_low": float(np.percentile(boot_diff_f1, 2.5)),
                "delta_macro_f1_ci95_high": float(np.percentile(boot_diff_f1, 97.5)),
                "test_accuracy_a": acc_a,
                "test_accuracy_b": acc_b,
                "delta_accuracy_a_minus_b": acc_a - acc_b,
                "delta_accuracy_ci95_low": float(np.percentile(boot_diff_acc, 2.5)),
                "delta_accuracy_ci95_high": float(np.percentile(boot_diff_acc, 97.5)),
                "a_only_correct": a_only_correct,
                "b_only_correct": b_only_correct,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
            }
        )

    return TestClassificationOutputs(
        summary=pd.DataFrame(summary_rows),
        per_category=pd.concat(per_cat_frames, ignore_index=True),
        paired_contrasts=pd.DataFrame(contrast_rows),
        predictions=pd.concat(pred_frames, ignore_index=True),
    )


def run_final_test_evaluation(
    project_root: str | Path,
    data_root_override: str | Path | None = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> FinalTestEvaluationOutputs:
    """Execute the complete one-time Stage 08 held-out test evaluation."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    selection_config = build_segmentation_selection_config(root)

    seg_outputs = evaluate_test_segmentation(
        root,
        data_root_override=data_root_override,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    cls_outputs = evaluate_test_classification(
        root,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )

    seg_macro_idx = seg_outputs.macro_summary.set_index("method")
    cls_sum_idx = cls_outputs.summary.set_index("model_id")

    manifest: dict[str, object] = {
        "schema_version": "final-test-evaluation-v1",
        "evaluation_split": "test",
        "split_sha256": split_sha,
        "cohort_counts": {
            "test_images": 319,
            "test_groups": 53,
            "categories": 6,
            "segmentation_methods_evaluated": len(SUPPORTED_METHODS),
            "classification_models_evaluated": int(len(cls_outputs.summary)),
        },
        "bootstrap_protocol": {
            "resampling_unit": "group_id (53 test groups sampled with replacement)",
            "n_resamples": int(bootstrap_resamples),
            "seed": int(seed),
            "confidence_level": 0.95,
        },
        "primary_segmentation_test_result": {
            "method": PRIMARY_FEATURE_METHOD,
            "val_mean_category_dice": float(
                seg_macro_idx.loc[PRIMARY_FEATURE_METHOD, "val_mean_category_dice"]
            ),
            "test_mean_category_dice": float(
                seg_macro_idx.loc[PRIMARY_FEATURE_METHOD, "test_mean_category_dice"]
            ),
            "test_dice_ci95": [
                float(seg_macro_idx.loc[PRIMARY_FEATURE_METHOD, "test_dice_ci95_low"]),
                float(seg_macro_idx.loc[PRIMARY_FEATURE_METHOD, "test_dice_ci95_high"]),
            ],
            "val_mean_category_iou": float(
                seg_macro_idx.loc[PRIMARY_FEATURE_METHOD, "val_mean_category_iou"]
            ),
            "test_mean_category_iou": float(
                seg_macro_idx.loc[PRIMARY_FEATURE_METHOD, "test_mean_category_iou"]
            ),
            "test_delta_dice_vs_control": float(
                seg_macro_idx.loc[PRIMARY_FEATURE_METHOD, "test_delta_dice_vs_control"]
            ),
        },
        "primary_classification_test_result": {
            "model_id": "svm_k4_M0_selected",
            "segmentation_method": PRIMARY_FEATURE_METHOD,
            "feature_set": "C",
            "val_macro_f1": float(cls_sum_idx.loc["svm_k4_M0_selected", "val_macro_f1"]),
            "test_macro_f1": float(cls_sum_idx.loc["svm_k4_M0_selected", "test_macro_f1"]),
            "test_macro_f1_ci95": [
                float(cls_sum_idx.loc["svm_k4_M0_selected", "test_macro_f1_ci95_low"]),
                float(cls_sum_idx.loc["svm_k4_M0_selected", "test_macro_f1_ci95_high"]),
            ],
            "val_balanced_accuracy": float(
                cls_sum_idx.loc["svm_k4_M0_selected", "val_balanced_accuracy"]
            ),
            "test_balanced_accuracy": float(
                cls_sum_idx.loc["svm_k4_M0_selected", "test_balanced_accuracy"]
            ),
            "val_accuracy": float(cls_sum_idx.loc["svm_k4_M0_selected", "val_accuracy"]),
            "test_accuracy": float(cls_sum_idx.loc["svm_k4_M0_selected", "test_accuracy"]),
        },
        "best_comparator_classification_test_result": {
            "model_id": "svm_otsu_M2_C",
            "segmentation_method": "otsu_M2",
            "feature_set": "C",
            "val_macro_f1": float(cls_sum_idx.loc["svm_otsu_M2_C", "val_macro_f1"]),
            "test_macro_f1": float(cls_sum_idx.loc["svm_otsu_M2_C", "test_macro_f1"]),
            "test_macro_f1_ci95": [
                float(cls_sum_idx.loc["svm_otsu_M2_C", "test_macro_f1_ci95_low"]),
                float(cls_sum_idx.loc["svm_otsu_M2_C", "test_macro_f1_ci95_high"]),
            ],
            "val_balanced_accuracy": float(
                cls_sum_idx.loc["svm_otsu_M2_C", "val_balanced_accuracy"]
            ),
            "test_balanced_accuracy": float(
                cls_sum_idx.loc["svm_otsu_M2_C", "test_balanced_accuracy"]
            ),
            "val_accuracy": float(cls_sum_idx.loc["svm_otsu_M2_C", "val_accuracy"]),
            "test_accuracy": float(cls_sum_idx.loc["svm_otsu_M2_C", "test_accuracy"]),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
            "pillow": pillow_version,
            "scikit_learn": sklearn.__version__,
        },
    }

    return FinalTestEvaluationOutputs(
        segmentation=seg_outputs,
        classification=cls_outputs,
        selection_config=selection_config,
        manifest=manifest,
        split_sha256=split_sha,
    )
