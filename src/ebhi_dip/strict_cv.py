"""Strict fold-aware end-to-end 5-fold CV and validation-blind selection for Stage 07 2x3 SVM."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedGroupKFold

from ebhi_dip.classification import (
    ALL_FEATURE_SETS,
    CLASSIFICATION_2X3_DIR_REL,
    COMPARISON_2X3_METHODS,
    CV_FOLDS,
    DECISION_PARITY_ATOL,
    DEFAULT_BASELINE_SPEC,
    PRIMARY_FEATURE_METHOD,
    SENSITIVITY_FEATURE_SETS,
    SHARED_APPEARANCE_METHOD,
    SVM_CLASS_WEIGHTS,
    SVM_C_VALUES,
    SVM_KERNELS,
    SVM_RBF_GAMMAS,
    _compute_seg_vs_cls_correlations,
    _format_gamma,
    _parse_gamma,
    _prediction_rows_for_model,
    apply_standard_scaler,
    classification_metrics,
    cv_selection_key_for_svm,
    export_svm_artifact,
    feature_columns_for_set,
    fit_standard_scaler,
    fit_svm_model,
    load_classification_feature_frame,
    load_shared_appearance_frame,
    per_category_classification_table,
    verify_svm_artifact_parity,
)
from ebhi_dip.dataset import (
    EXPECTED_CATEGORIES,
    REFERENCE_THRESHOLD,
    load_raw_mask,
    load_reference_mask,
    load_rgb,
    load_split_manifest,
)
from ebhi_dip.evaluation import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    _group_bootstrap_indices,
    load_validation_results,
    summarize_validation_methods,
)
from ebhi_dip.features import (
    APPEARANCE_FEATURE_NAMES,
    MORPHOLOGY_FEATURE_NAMES,
    SEGMENTED_APPEARANCE_FEATURE_NAMES,
    _appearance_from_lab_and_glcm,
    _glcm_properties,
    _masked_glcm,
    _morphology_payload,
    _quantize_gray,
)
from ebhi_dip.metrics import category_macro_scores, overlap_scores
from ebhi_dip.paths import resolve_manifest_path, verify_frozen_split
from ebhi_dip.provenance import sha256_bytes, sha256_source_text
from ebhi_dip.segmentation import (
    load_kmeans_model,
    load_otsu_config,
    morphology_variants,
    nearest_clusters,
    otsu_threshold_mask,
    rgb_to_gray,
    rgb_to_lab,
)

CLASSIFICATION_2X3_STRICT_DIR_REL = Path("results/classification_2x3_strict")
MODELS_CV_2X3_STRICT_DIR_REL = Path("models/cv_2x3_strict")
STRICT_CV_CACHE_DIR_REL = Path("data/processed/strict_cv_2x3")

KMEANS_PIXELS_PER_IMAGE = 1000
KMEANS_SAMPLE_SEED = 42
FOREGROUND_RATE_THRESHOLD = 0.5

SHARED_KMEANS_PARAMETERS: dict[str, object] = {
    "init": "k-means++",
    "batch_size": 4096,
    "init_size": 12288,
    "n_init": 10,
    "random_state": 42,
    "max_iter": 100,
    "max_no_improvement": 20,
    "tol": 0.0,
    "reassignment_ratio": 0.01,
    "compute_labels": True,
}

STRICT_KEY_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    ("otsu_M0_C_vs_shared_A", "svm_otsu_M0_C", "svm_shared_A", "Raw Otsu Set C vs Shared Set A"),
    ("otsu_M0_B_vs_shared_A", "svm_otsu_M0_B", "svm_shared_A", "Pure raw Otsu morphology Set B vs Shared Set A"),
    ("otsu_M0_C_vs_otsu_M0_B", "svm_otsu_M0_C", "svm_otsu_M0_B", "Combining Set A + raw Otsu morphology vs morphology alone"),
    ("otsu_M0_C_vs_otsu_M1_C", "svm_otsu_M0_C", "svm_otsu_M1_C", "Otsu M0 vs M1 (>=16 px removal) on Set C"),
    ("otsu_M0_C_vs_otsu_M2_C", "svm_otsu_M0_C", "svm_otsu_M2_C", "Otsu M0 vs M2 (3x3 cross open/close) on Set C"),
    ("otsu_M1_C_vs_otsu_M2_C", "svm_otsu_M1_C", "svm_otsu_M2_C", "Otsu M1 vs M2 on Set C"),
    ("otsu_M0_C_vs_k4_M0_C", "svm_otsu_M0_C", "svm_k4_M0_C", "Raw Otsu vs raw K-Means k=4 on Set C"),
    ("otsu_M2_C_vs_k4_M0_C", "svm_otsu_M2_C", "svm_k4_M0_C", "Smoothed Otsu M2 vs raw K-Means k=4 on Set C"),
    ("otsu_M0_B_vs_k4_M0_B", "svm_otsu_M0_B", "svm_k4_M0_B", "Pure morphology Set B: Otsu M0 vs K-Means k4 M0"),
    ("k4_M0_C_vs_k4_M1_C", "svm_k4_M0_C", "svm_k4_M1_C", "K-Means k=4 M0 vs M1 on Set C"),
    ("k4_M0_C_vs_k4_M2_C", "svm_k4_M0_C", "svm_k4_M2_C", "K-Means k=4 M0 vs M2 on Set C"),
    ("k4_M0_C_vs_shared_A", "svm_k4_M0_C", "svm_shared_A", "K-Means k4 M0 Set C vs Shared Set A"),
    ("k4_M0_B_vs_shared_A", "svm_k4_M0_B", "svm_shared_A", "K-Means k4 M0 morphology Set B vs Shared Set A"),
    ("k4_M0_C_seg_vs_k4_M0_C", "svm_k4_M0_C_seg", "svm_k4_M0_C", "Masked foreground appearance C_seg vs unmasked Set C on k4_M0"),
    ("k4_M0_A_seg_vs_shared_A", "svm_k4_M0_A_seg", "svm_shared_A", "Masked foreground appearance A_seg vs global Set A on k4_M0"),
)

_STRICT_WORKER_STATE: dict[str, object] = {}


@dataclass(frozen=True)
class Svm2x3StrictExperimentOutputs:
    """Outputs of the strict fold-aware, validation-blind Stage 07 2x3 SVM protocol."""

    grid_results: pd.DataFrame
    selected_summary: pd.DataFrame
    per_category_val: pd.DataFrame
    segmentation_vs_classification: pd.DataFrame
    val_predictions: pd.DataFrame
    val_bootstrap_models: pd.DataFrame
    val_paired_contrasts: pd.DataFrame
    val_bootstrap_all6_models: pd.DataFrame
    val_paired_contrasts_all6: pd.DataFrame
    bootstrap_sensitivity_comparison: pd.DataFrame
    bootstrap_sensitivity_summary: dict[str, object]
    protocol_comparison: pd.DataFrame
    fold_upstream_audit: dict[str, object]
    model_artifacts: dict[str, dict[str, object]]
    manifest: dict[str, object]
    split_sha256: str


def _present_categories_ordered(categories: Sequence[str]) -> tuple[str, ...]:
    cat_set = set(str(c) for c in categories)
    ordered = [c for c in EXPECTED_CATEGORIES if c in cat_set]
    extras = sorted(c for c in cat_set if c not in EXPECTED_CATEGORIES)
    return tuple(ordered + extras)


def _sample_ids_hash(sample_ids: Sequence[str]) -> str:
    joined = "\n".join(sorted(str(sid) for sid in sample_ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def compute_otsu_polarity_dice_table(
    records_df: pd.DataFrame,
    project_root: str | Path,
    data_root_override: str | Path | None = None,
) -> pd.DataFrame:
    """Evaluate per-image dark vs bright Otsu Dice against reference masks."""
    root = Path(project_root).expanduser().resolve()
    rows: list[dict[str, object]] = []
    ordered = records_df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    for _, rec in ordered.iterrows():
        img = load_rgb(resolve_manifest_path(root, rec["image_path"], data_root_override=data_root_override))
        ref = load_reference_mask(
            resolve_manifest_path(root, rec["mask_path"], data_root_override=data_root_override),
            threshold=REFERENCE_THRESHOLD,
        )
        _, dark_mask = otsu_threshold_mask(img, foreground_polarity="dark")
        bright_mask = ~dark_mask
        dark_dice, dark_iou = overlap_scores(dark_mask, ref)
        bright_dice, bright_iou = overlap_scores(bright_mask, ref)
        rows.append(
            {
                "sample_id": str(rec["sample_id"]),
                "category": str(rec["category"]),
                "group_id": str(rec["group_id"]),
                "dark_dice": float(dark_dice),
                "dark_iou": float(dark_iou),
                "bright_dice": float(bright_dice),
                "bright_iou": float(bright_iou),
            }
        )
    return pd.DataFrame(rows)


def fit_fold_otsu_polarity(
    fold_train_df: pd.DataFrame,
    project_root: str | Path,
    data_root_override: str | Path | None = None,
    precomputed_dice_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Select global Otsu foreground polarity strictly from fold-training reference masks."""
    if len(fold_train_df) == 0:
        raise ValueError("fold_train_df cannot be empty when fitting Otsu polarity.")

    train_ids = set(fold_train_df["sample_id"].astype(str))
    if precomputed_dice_df is not None:
        sub_dice = precomputed_dice_df.loc[precomputed_dice_df["sample_id"].astype(str).isin(train_ids)].copy()
        if len(sub_dice) != len(fold_train_df):
            raise ValueError("precomputed_dice_df does not cover every fold_train_df sample_id.")
    else:
        sub_dice = compute_otsu_polarity_dice_table(
            fold_train_df,
            project_root=project_root,
            data_root_override=data_root_override,
        )

    long_rows: list[dict[str, object]] = []
    for _, rec in sub_dice.iterrows():
        for pol in ("dark", "bright"):
            long_rows.append(
                {
                    "sample_id": str(rec["sample_id"]),
                    "category": str(rec["category"]),
                    "method": pol,
                    "dice": float(rec[f"{pol}_dice"]),
                    "iou": float(rec[f"{pol}_iou"]),
                }
            )
    expected_cats = _present_categories_ordered(sub_dice["category"].tolist())
    macro = category_macro_scores(pd.DataFrame(long_rows), expected_categories=expected_cats)
    dark_dice = float(macro.loc["dark", "mean_category_dice"])
    bright_dice = float(macro.loc["bright", "mean_category_dice"])
    chosen = "dark" if dark_dice >= bright_dice else "bright"
    return {
        "chosen_polarity": chosen,
        "dark_mean_category_dice": dark_dice,
        "bright_mean_category_dice": bright_dice,
        "margin_dark_minus_bright": float(dark_dice - bright_dice),
        "n_train_images": int(len(fold_train_df)),
        "n_train_groups": int(fold_train_df["group_id"].nunique()),
        "train_sample_ids_sha256": _sample_ids_hash(fold_train_df["sample_id"].astype(str).tolist()),
    }


def fit_fold_kmeans_k4_mapped(
    fold_train_df: pd.DataFrame,
    project_root: str | Path,
    data_root_override: str | Path | None = None,
    k: int = 4,
    pixels_per_image: int = KMEANS_PIXELS_PER_IMAGE,
    seed: int = KMEANS_SAMPLE_SEED,
    foreground_rate_threshold: float = FOREGROUND_RATE_THRESHOLD,
    _preloaded_assets: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, object]:
    """Sample Lab pixels, fit MiniBatchKMeans, and map foreground clusters strictly on fold_train_df."""
    if len(fold_train_df) == 0:
        raise ValueError("fold_train_df cannot be empty when fitting fold K-Means.")

    root = Path(project_root).expanduser().resolve()
    ordered = fold_train_df.sort_values("sample_id", ascending=True, kind="mergesort").reset_index(drop=True)
    rng = np.random.default_rng(int(seed))

    lab_parts: list[np.ndarray] = []
    ref_parts: list[np.ndarray] = []
    for _, rec in ordered.iterrows():
        sid = str(rec["sample_id"])
        if _preloaded_assets is not None and sid in _preloaded_assets:
            img, ref_2d = _preloaded_assets[sid]
        else:
            img = load_rgb(resolve_manifest_path(root, rec["image_path"], data_root_override=data_root_override))
            raw_mask = load_raw_mask(resolve_manifest_path(root, rec["mask_path"], data_root_override=data_root_override))
            if raw_mask.shape != img.shape[:2]:
                raise ValueError(f"Mask/image shape mismatch for sample {sid}: {raw_mask.shape} vs {img.shape[:2]}.")
            ref_2d = raw_mask >= REFERENCE_THRESHOLD

        if img.ndim != 3 or img.shape[2] != 3 or ref_2d.shape != img.shape[:2]:
            raise ValueError(f"Unexpected dimensions for sample {sid}.")
        lab = rgb_to_lab(img).reshape(-1, 3)
        ref = np.asarray(ref_2d, dtype=bool).reshape(-1)
        n_pick = min(int(pixels_per_image), int(lab.shape[0]))
        indices = rng.choice(lab.shape[0], size=n_pick, replace=False)
        lab_parts.append(lab[indices])
        ref_parts.append(ref[indices])

    x_lab = np.ascontiguousarray(np.concatenate(lab_parts, axis=0), dtype=np.float32)
    y_ref = np.concatenate(ref_parts, axis=0)
    if not np.isfinite(x_lab).all():
        raise ValueError("Sampled Lab pixels contain non-finite values.")

    params = dict(SHARED_KMEANS_PARAMETERS)
    if len(x_lab) < int(params["init_size"]):
        params["init_size"] = int(len(x_lab))
    if len(x_lab) < int(params["batch_size"]):
        params["batch_size"] = int(len(x_lab))

    model = MiniBatchKMeans(n_clusters=int(k), **params)
    model.fit(x_lab)
    centers = model.cluster_centers_

    labels = nearest_clusters(x_lab, centers)
    counts = np.bincount(labels, minlength=int(k))
    if np.any(counts == 0):
        raise RuntimeError(f"Fold K-Means k={k} encountered an empty cluster.")

    fg_counts = np.bincount(labels[y_ref], minlength=int(k))
    rates = fg_counts / counts
    is_fg = rates >= float(foreground_rate_threshold)
    fg_clusters = np.flatnonzero(is_fg).astype(int).tolist()
    if not fg_clusters:
        raise RuntimeError(f"Fold K-Means k={k} mapped zero foreground clusters at threshold={foreground_rate_threshold}.")

    return {
        "k": int(k),
        "cluster_centers": centers.tolist(),
        "cluster_centers_dtype": str(centers.dtype),
        "foreground_clusters": fg_clusters,
        "foreground_rate_threshold": float(foreground_rate_threshold),
        "cluster_pixel_counts": counts.astype(int).tolist(),
        "cluster_reference_foreground_counts": fg_counts.astype(int).tolist(),
        "cluster_reference_foreground_rates": [float(r) for r in rates.tolist()],
        "inertia": float(model.inertia_),
        "training_images": int(len(ordered)),
        "training_groups": int(ordered["group_id"].nunique()),
        "training_pixel_count": int(len(x_lab)),
        "pixels_per_image": int(pixels_per_image),
        "seed": int(seed),
        "train_sample_ids_sha256": _sample_ids_hash(ordered["sample_id"].astype(str).tolist()),
    }


def _masked_appearance_from_precomputed(
    lab: np.ndarray,
    quantized: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Exact equivalent to extract_masked_appearance_features using precomputed Lab/quantized arrays."""
    features = {name: 0.0 for name in SEGMENTED_APPEARANCE_FEATURE_NAMES}
    if not mask.any():
        return features
    foreground_lab = lab[mask].astype(np.float64)
    matrix = _masked_glcm(quantized, mask)
    glcm_features = _glcm_properties(matrix)
    base = _appearance_from_lab_and_glcm(foreground_lab, glcm_features)
    for name, value in base.items():
        features[f"seg_{name}"] = float(value)
    return features


def _init_strict_worker(
    project_root: str,
    data_root_override: str | None,
    fold_k4_models: list[dict[str, object]],
    fold_otsu_polarities: list[str],
    global_k4_model: dict[str, object],
    methods: tuple[str, ...],
) -> None:
    _STRICT_WORKER_STATE.clear()
    _STRICT_WORKER_STATE["project_root"] = Path(project_root)
    _STRICT_WORKER_STATE["data_root_override"] = Path(data_root_override) if data_root_override else None
    _STRICT_WORKER_STATE["fold_k4_models"] = fold_k4_models
    _STRICT_WORKER_STATE["fold_otsu_polarities"] = fold_otsu_polarities
    _STRICT_WORKER_STATE["global_k4_model"] = global_k4_model
    _STRICT_WORKER_STATE["methods"] = methods


def _extract_single_image_all_folds(
    record: dict[str, object],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute fold-specific morphology & masked-appearance features for one train image across all folds."""
    state = _STRICT_WORKER_STATE
    root: Path = state["project_root"]  # type: ignore[assignment]
    data_root: Path | None = state["data_root_override"]  # type: ignore[assignment]
    fold_k4_models: list[dict[str, object]] = state["fold_k4_models"]  # type: ignore[assignment]
    fold_otsu_polarities: list[str] = state["fold_otsu_polarities"]  # type: ignore[assignment]
    global_k4_model: dict[str, object] = state["global_k4_model"]  # type: ignore[assignment]
    methods: tuple[str, ...] = state["methods"]  # type: ignore[assignment]

    img = load_rgb(resolve_manifest_path(root, str(record["image_path"]), data_root_override=data_root))
    lab = rgb_to_lab(img)
    flat_lab = lab.reshape(-1, 3)
    quantized = _quantize_gray(rgb_to_gray(img))
    n_folds = len(fold_k4_models)
    n_methods = len(methods)

    morph_arr = np.zeros((n_folds, n_methods, len(MORPHOLOGY_FEATURE_NAMES)), dtype=np.float64)
    seg_app_arr = np.zeros((n_folds, len(SEGMENTED_APPEARANCE_FEATURE_NAMES)), dtype=np.float64)
    k4_m0_agreement = np.zeros(n_folds, dtype=np.float64)
    k4_m0_iou_vs_full = np.zeros(n_folds, dtype=np.float64)

    method_idx = {m: i for i, m in enumerate(methods)}
    morph_cache: dict[bytes, np.ndarray] = {}
    seg_app_cache: dict[bytes, np.ndarray] = {}

    def _get_morph_vec(mask: np.ndarray) -> np.ndarray:
        key = mask.tobytes()
        cached = morph_cache.get(key)
        if cached is not None:
            return cached
        mp, _ = _morphology_payload(mask)
        vec = np.asarray([float(mp[col]) for col in MORPHOLOGY_FEATURE_NAMES], dtype=np.float64)
        morph_cache[key] = vec
        return vec

    def _get_seg_app_vec(mask: np.ndarray) -> np.ndarray:
        key = mask.tobytes()
        cached = seg_app_cache.get(key)
        if cached is not None:
            return cached
        sa = _masked_appearance_from_precomputed(lab, quantized, mask)
        vec = np.asarray([float(sa[col]) for col in SEGMENTED_APPEARANCE_FEATURE_NAMES], dtype=np.float64)
        seg_app_cache[key] = vec
        return vec

    # Reference full-train k4_M0 mask to quantify fold centroid/mapping stability
    ref_labels = nearest_clusters(flat_lab, global_k4_model["cluster_centers"]).reshape(img.shape[:2])
    ref_k4_m0 = np.isin(ref_labels, global_k4_model["foreground_clusters"])

    # Compute Otsu once per unique polarity selected across folds (typically only {"dark"})
    otsu_by_polarity: dict[str, dict[str, np.ndarray]] = {}
    for pol in sorted(set(fold_otsu_polarities)):
        _, otsu_raw = otsu_threshold_mask(img, foreground_polarity=pol)
        otsu_vars = morphology_variants(otsu_raw)
        otsu_by_polarity[pol] = {
            f"otsu_{stage}": _get_morph_vec(otsu_vars[stage]) for stage in ("M0", "M1", "M2")
        }

    for fold_idx in range(n_folds):
        pol = fold_otsu_polarities[fold_idx]
        for stage in ("M0", "M1", "M2"):
            m_name = f"otsu_{stage}"
            if m_name in method_idx:
                morph_arr[fold_idx, method_idx[m_name]] = otsu_by_polarity[pol][m_name]

        km_model = fold_k4_models[fold_idx]
        lbl = nearest_clusters(flat_lab, km_model["cluster_centers"]).reshape(img.shape[:2])
        km_raw = np.isin(lbl, km_model["foreground_clusters"])
        km_vars = morphology_variants(km_raw)

        for stage in ("M0", "M1", "M2"):
            m_name = f"k4_{stage}"
            if m_name in method_idx:
                morph_arr[fold_idx, method_idx[m_name]] = _get_morph_vec(km_vars[stage])

        seg_app_arr[fold_idx] = _get_seg_app_vec(km_vars["M0"])
        k4_m0 = km_vars["M0"]
        k4_m0_agreement[fold_idx] = float(np.mean(k4_m0 == ref_k4_m0))
        _, iou_full = overlap_scores(k4_m0, ref_k4_m0)
        k4_m0_iou_vs_full[fold_idx] = float(iou_full)

    return str(record["sample_id"]), morph_arr, seg_app_arr, k4_m0_agreement, k4_m0_iou_vs_full


def _preload_train_assets_and_otsu(
    train_meta_df: pd.DataFrame,
    project_root: Path,
    data_root_override: Path | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Load training RGB/reference masks once and compute per-image dark/bright Otsu Dice scores."""
    assets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    ordered = train_meta_df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    for _, rec in ordered.iterrows():
        sid = str(rec["sample_id"])
        img = load_rgb(resolve_manifest_path(project_root, str(rec["image_path"]), data_root_override=data_root_override))
        ref = load_reference_mask(
            resolve_manifest_path(project_root, str(rec["mask_path"]), data_root_override=data_root_override),
            threshold=REFERENCE_THRESHOLD,
        )
        assets[sid] = (img, ref)
        _, dark_mask = otsu_threshold_mask(img, foreground_polarity="dark")
        bright_mask = ~dark_mask
        dark_dice, dark_iou = overlap_scores(dark_mask, ref)
        bright_dice, bright_iou = overlap_scores(bright_mask, ref)
        rows.append(
            {
                "sample_id": sid,
                "category": str(rec["category"]),
                "group_id": str(rec["group_id"]),
                "dark_dice": float(dark_dice),
                "dark_iou": float(dark_iou),
                "bright_dice": float(bright_dice),
                "bright_iou": float(bright_iou),
            }
        )
    return assets, pd.DataFrame(rows)


def _build_fold_feature_cache(
    project_root: Path,
    train_meta_df: pd.DataFrame,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    methods: tuple[str, ...],
    global_otsu_polarity: str,
    global_k4_model: dict[str, object],
    split_sha: str,
    data_root_override: Path | None = None,
    workers: int = 1,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Fit strict per-fold upstream models and compute or load cached fold feature arrays."""
    preloaded_assets, precomputed_otsu_dice = _preload_train_assets_and_otsu(
        train_meta_df,
        project_root=project_root,
        data_root_override=data_root_override,
    )

    fold_otsu_info: list[dict[str, object]] = []
    fold_k4_info: list[dict[str, object]] = []
    for fold_idx, (f_tr_idx, f_va_idx) in enumerate(fold_splits):
        fold_train_df = train_meta_df.iloc[f_tr_idx].copy()
        fold_val_df = train_meta_df.iloc[f_va_idx].copy()
        tr_ids = set(fold_train_df["sample_id"].astype(str))
        va_ids = set(fold_val_df["sample_id"].astype(str))
        tr_groups = set(fold_train_df["group_id"].astype(str))
        va_groups = set(fold_val_df["group_id"].astype(str))
        if not tr_ids.isdisjoint(va_ids):
            raise ValueError(f"Fold {fold_idx}: train and val sample_ids overlap.")
        if not tr_groups.isdisjoint(va_groups):
            raise ValueError(f"Fold {fold_idx}: train and val group_ids overlap.")

        otsu_res = fit_fold_otsu_polarity(
            fold_train_df,
            project_root=project_root,
            data_root_override=data_root_override,
            precomputed_dice_df=precomputed_otsu_dice,
        )
        k4_res = fit_fold_kmeans_k4_mapped(
            fold_train_df,
            project_root=project_root,
            data_root_override=data_root_override,
            k=4,
            _preloaded_assets=preloaded_assets,
        )
        fold_otsu_info.append(otsu_res)
        fold_k4_info.append(k4_res)
    preloaded_assets.clear()

    fold_otsu_polarities = [str(item["chosen_polarity"]) for item in fold_otsu_info]
    sig_payload = {
        "split_sha256": split_sha,
        "methods": list(methods),
        "sample_ids": train_meta_df["sample_id"].astype(str).tolist(),
        "fold_otsu": fold_otsu_polarities,
        "fold_k4": [
            {
                "centers": m["cluster_centers"],
                "fg": m["foreground_clusters"],
                "train_hash": m["train_sample_ids_sha256"],
            }
            for m in fold_k4_info
        ],
        "segmentation_sha": sha256_source_text(project_root / "src/ebhi_dip/segmentation.py"),
        "features_sha": sha256_source_text(project_root / "src/ebhi_dip/features.py"),
    }
    cache_key = hashlib.sha256(
        json.dumps(sig_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path = project_root / STRICT_CV_CACHE_DIR_REL / "strict_fold_cache.npz"

    n_splits = len(fold_splits)
    n_train = len(train_meta_df)
    n_methods = len(methods)

    loaded_from_cache = False
    if use_cache and data_root_override is None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                str(cached["cache_key"].item()) == cache_key
                and cached["sample_ids"].astype(str).tolist() == train_meta_df["sample_id"].astype(str).tolist()
            ):
                fold_morph = np.array(cached["fold_morph"], dtype=np.float64)
                fold_seg_app = np.array(cached["fold_seg_app"], dtype=np.float64)
                k4_agreement = np.array(cached["k4_m0_agreement"], dtype=np.float64)
                k4_iou_full = np.array(cached["k4_m0_iou_vs_full"], dtype=np.float64)
                loaded_from_cache = True

    if not loaded_from_cache:
        records = [
            {
                "sample_id": str(row["sample_id"]),
                "image_path": str(row["image_path"]),
            }
            for _, row in train_meta_df.iterrows()
        ]
        fold_morph = np.zeros((n_splits, n_train, n_methods, len(MORPHOLOGY_FEATURE_NAMES)), dtype=np.float64)
        fold_seg_app = np.zeros((n_splits, n_train, len(SEGMENTED_APPEARANCE_FEATURE_NAMES)), dtype=np.float64)
        k4_agreement = np.zeros((n_splits, n_train), dtype=np.float64)
        k4_iou_full = np.zeros((n_splits, n_train), dtype=np.float64)

        id_to_pos = {str(sid): idx for idx, sid in enumerate(train_meta_df["sample_id"].astype(str))}
        data_override_str = str(data_root_override) if data_root_override is not None else None
        max_workers = max(1, int(workers))

        if max_workers == 1:
            _init_strict_worker(
                str(project_root),
                data_override_str,
                fold_k4_info,
                fold_otsu_polarities,
                global_k4_model,
                methods,
            )
            results_iter = map(_extract_single_image_all_folds, records)
            for sid, m_arr, sa_arr, ag_arr, iou_arr in results_iter:
                pos = id_to_pos[sid]
                fold_morph[:, pos, :, :] = m_arr
                fold_seg_app[:, pos, :] = sa_arr
                k4_agreement[:, pos] = ag_arr
                k4_iou_full[:, pos] = iou_arr
        else:
            chunksize = max(1, min(32, len(records) // (max_workers * 4) or 1))
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_strict_worker,
                initargs=(
                    str(project_root),
                    data_override_str,
                    fold_k4_info,
                    fold_otsu_polarities,
                    global_k4_model,
                    methods,
                ),
            ) as executor:
                for sid, m_arr, sa_arr, ag_arr, iou_arr in executor.map(
                    _extract_single_image_all_folds, records, chunksize=chunksize
                ):
                    pos = id_to_pos[sid]
                    fold_morph[:, pos, :, :] = m_arr
                    fold_seg_app[:, pos, :] = sa_arr
                    k4_agreement[:, pos] = ag_arr
                    k4_iou_full[:, pos] = iou_arr

        if use_cache and data_root_override is None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                cache_key=cache_key,
                sample_ids=np.asarray(train_meta_df["sample_id"].astype(str).tolist(), dtype=str),
                methods=np.asarray([str(m) for m in methods], dtype=str),
                fold_morph=fold_morph,
                fold_seg_app=fold_seg_app,
                k4_m0_agreement=k4_agreement,
                k4_m0_iou_vs_full=k4_iou_full,
            )

    global_centers = np.asarray(global_k4_model["cluster_centers"], dtype=np.float64)
    fold_audit: list[dict[str, object]] = []
    for fold_idx, (f_tr_idx, f_va_idx) in enumerate(fold_splits):
        fold_centers = np.asarray(fold_k4_info[fold_idx]["cluster_centers"], dtype=np.float64)
        # Match fold clusters to closest global clusters via minimum L2 distance sum
        from scipy.optimize import linear_sum_assignment

        dist_mat = np.linalg.norm(fold_centers[:, None, :] - global_centers[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(dist_mat)
        matched_center_l2 = [float(dist_mat[r, c]) for r, c in zip(row_ind, col_ind)]

        fold_audit.append(
            {
                "fold": int(fold_idx),
                "fold_train_images": int(len(f_tr_idx)),
                "fold_val_images": int(len(f_va_idx)),
                "fold_train_groups": int(train_meta_df.iloc[f_tr_idx]["group_id"].nunique()),
                "fold_val_groups": int(train_meta_df.iloc[f_va_idx]["group_id"].nunique()),
                "group_id_disjoint": True,
                "sample_id_disjoint": True,
                "otsu_polarity_fit": fold_otsu_info[fold_idx],
                "otsu_matches_full_train_polarity": fold_otsu_info[fold_idx]["chosen_polarity"] == global_otsu_polarity,
                "k4_fit": fold_k4_info[fold_idx],
                "k4_cluster_center_matched_l2_vs_full_train": matched_center_l2,
                "k4_cluster_center_mean_l2_vs_full_train": float(np.mean(matched_center_l2)),
                "k4_m0_val_fold_pixel_agreement_mean": float(np.mean(k4_agreement[fold_idx, f_va_idx])),
                "k4_m0_val_fold_pixel_agreement_min": float(np.min(k4_agreement[fold_idx, f_va_idx])),
                "k4_m0_val_fold_iou_vs_full_train_mean": float(np.mean(k4_iou_full[fold_idx, f_va_idx])),
                "k4_m0_val_fold_iou_vs_full_train_min": float(np.min(k4_iou_full[fold_idx, f_va_idx])),
                "loaded_from_cache": loaded_from_cache,
            }
        )

    return fold_morph, fold_seg_app, fold_audit


def evaluate_strict_cv_grid(
    y_train: np.ndarray,
    fold_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    fold_feature_matrices: Sequence[np.ndarray],
    feature_names: Sequence[str],
    method: str,
    feature_set: str,
    track: str,
    selection_key: Callable[[Mapping[str, object]], tuple[object, ...]] = cv_selection_key_for_svm,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """Run validation-blind 5-fold CV grid search with fold-internal features and scalers."""
    if len(fold_splits) != len(fold_feature_matrices):
        raise ValueError("fold_splits and fold_feature_matrices must have equal length.")

    cols = list(feature_names)
    fold_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_idx, (f_tr_idx, f_va_idx) in enumerate(fold_splits):
        X_k = fold_feature_matrices[fold_idx]
        if X_k.shape[1] != len(cols):
            raise ValueError(f"Fold {fold_idx} width {X_k.shape[1]} != {len(cols)} for {method}/{feature_set}.")
        sc_fold = fit_standard_scaler(X_k[f_tr_idx], cols)
        fold_cache.append(
            (
                f_va_idx,
                apply_standard_scaler(X_k[f_tr_idx], sc_fold),
                y_train[f_tr_idx],
                apply_standard_scaler(X_k[f_va_idx], sc_fold),
            )
        )

    rows: list[dict[str, object]] = []
    for kernel in SVM_KERNELS:
        gammas: Sequence[str | float] = ("scale",) if kernel == "linear" else SVM_RBF_GAMMAS
        for cw in SVM_CLASS_WEIGHTS:
            for C in SVM_C_VALUES:
                for gamma in gammas:
                    gamma_str = _format_gamma(gamma)
                    oof_pred = np.empty_like(y_train)
                    fold_macro_f1s: list[float] = []
                    fold_bal_accs: list[float] = []
                    fold_support_counts: list[int] = []
                    for f_va_idx, Xf_tr, yf_tr, Xf_va in fold_cache:
                        clf_fold = fit_svm_model(
                            Xf_tr, yf_tr, kernel=kernel, C=C, gamma=gamma, class_weight=cw
                        )
                        fold_pred = clf_fold.predict(Xf_va)
                        oof_pred[f_va_idx] = fold_pred
                        fold_m = classification_metrics(y_train[f_va_idx], fold_pred)
                        fold_macro_f1s.append(fold_m["macro_f1"])
                        fold_bal_accs.append(fold_m["balanced_accuracy"])
                        fold_support_counts.append(int(np.sum(clf_fold.n_support_)))

                    cv_scores = classification_metrics(y_train, oof_pred)
                    rows.append(
                        {
                            "track": track,
                            "method": method,
                            "feature_set": feature_set,
                            "n_features": len(cols),
                            "kernel": kernel,
                            "class_weight": cw,
                            "C": float(C),
                            "gamma": gamma_str,
                            "cv_macro_f1": cv_scores["macro_f1"],
                            "cv_fold_macro_f1_mean": float(np.mean(fold_macro_f1s)),
                            "cv_fold_macro_f1_std": float(np.std(fold_macro_f1s, ddof=0)),
                            "cv_fold_macro_f1_min": float(np.min(fold_macro_f1s)),
                            "cv_fold_macro_f1_max": float(np.max(fold_macro_f1s)),
                            "cv_balanced_accuracy": cv_scores["balanced_accuracy"],
                            "cv_fold_balanced_accuracy_mean": float(np.mean(fold_bal_accs)),
                            "cv_accuracy": cv_scores["accuracy"],
                            "cv_weighted_f1": cv_scores["weighted_f1"],
                            "cv_n_support_mean": float(np.mean(fold_support_counts)),
                        }
                    )

    best_row = max(rows, key=selection_key)
    default_row = next(
        r
        for r in rows
        if r["kernel"] == DEFAULT_BASELINE_SPEC["kernel"]
        and float(r["C"]) == float(DEFAULT_BASELINE_SPEC["C"])
        and r["gamma"] == str(DEFAULT_BASELINE_SPEC["gamma"])
        and r["class_weight"] == str(DEFAULT_BASELINE_SPEC["class_weight"])
    )

    grid_df = pd.DataFrame(rows)
    grid_df["is_default_baseline"] = (
        grid_df["kernel"].eq(default_row["kernel"])
        & grid_df["C"].eq(default_row["C"])
        & grid_df["gamma"].eq(default_row["gamma"])
        & grid_df["class_weight"].eq(default_row["class_weight"])
    )
    grid_df["is_selected_best"] = (
        grid_df["kernel"].eq(best_row["kernel"])
        & grid_df["C"].eq(best_row["C"])
        & grid_df["gamma"].eq(best_row["gamma"])
        & grid_df["class_weight"].eq(best_row["class_weight"])
    )
    if any(str(col).startswith("val_") for col in grid_df.columns):
        raise AssertionError("Validation metrics must never appear in strict CV grid search output.")
    return grid_df, dict(best_row), dict(default_row)


def _fast_six_class_bootstrap_metrics(
    yt_idx: np.ndarray,
    yp_idx: np.ndarray,
    boot_indices: Sequence[np.ndarray],
    n_classes: int = len(EXPECTED_CATEGORIES),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute bootstrap Macro-F1 (present categories), Balanced Accuracy, and Accuracy via 6x6 bincount."""
    n_boot = len(boot_indices)
    macro_f1 = np.zeros(n_boot, dtype=np.float64)
    bal_acc = np.zeros(n_boot, dtype=np.float64)
    acc = np.zeros(n_boot, dtype=np.float64)
    n_present = np.zeros(n_boot, dtype=np.int64)

    for b_idx, rows in enumerate(boot_indices):
        yt_b = yt_idx[rows]
        yp_b = yp_idx[rows]
        cm = np.bincount(yt_b * n_classes + yp_b, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        support = cm.sum(axis=1)
        pred_sum = cm.sum(axis=0)
        tp = np.diag(cm)
        present = support > 0
        n_present[b_idx] = int(np.sum(present))
        recalls = tp[present] / support[present]
        denom = support[present] + pred_sum[present]
        f1s = np.where(denom > 0, (2.0 * tp[present]) / denom, 0.0)
        macro_f1[b_idx] = float(np.mean(f1s))
        bal_acc[b_idx] = float(np.mean(recalls))
        acc[b_idx] = float(np.sum(tp) / len(rows))
    return macro_f1, bal_acc, acc, n_present


def _prepare_val_bootstrap_inputs(
    val_predictions_df: pd.DataFrame,
    v2_predictions_df: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
    pd.DataFrame | None,
]:
    """Validate and align validation predictions by sample_id for paired group bootstrap."""
    sel_df = val_predictions_df.loc[val_predictions_df["model_variant"].eq("cv_selected")].copy()
    cat_to_idx = {cat: idx for idx, cat in enumerate(EXPECTED_CATEGORIES)}
    pivot_pred = sel_df.pivot(index="sample_id", columns="model_id", values="predicted_category").sort_index()
    meta = (
        sel_df[["sample_id", "category", "group_id"]]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
        .loc[pivot_pred.index]
    )
    yt_str = meta["category"].astype(str).to_numpy()
    yt_idx = np.asarray([cat_to_idx[c] for c in yt_str], dtype=np.int64)
    groups = meta["group_id"].astype(str).to_numpy()

    v2_pivot: pd.DataFrame | None = None
    if v2_predictions_df is not None and not v2_predictions_df.empty:
        v2_sel = v2_predictions_df.loc[v2_predictions_df["model_variant"].eq("cv_selected")].copy()
        v2_pivot = v2_sel.pivot(index="sample_id", columns="model_id", values="predicted_category").reindex(
            pivot_pred.index
        )
    return sel_df, pivot_pred, meta, yt_str, yt_idx, groups, cat_to_idx, v2_pivot


def _paired_bootstrap_tables_from_indices(
    sel_df: pd.DataFrame,
    pivot_pred: pd.DataFrame,
    meta: pd.DataFrame,
    yt_str: np.ndarray,
    yt_idx: np.ndarray,
    groups: np.ndarray,
    cat_to_idx: Mapping[str, int],
    boot_indices: Sequence[np.ndarray],
    contrasts: Sequence[tuple[str, str, str, str]],
    seed: int,
    *,
    v2_pivot: pd.DataFrame | None = None,
    prop_gt_zero_col: str = "prob_delta_macro_f1_gt_0",
    extra_model_meta: Mapping[str, object] | None = None,
    extra_contrast_meta: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired group-blocked bootstrap model CIs and contrast CIs from precomputed index draws."""
    n_resamples = len(boot_indices)
    boot_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    point_cache: dict[str, dict[str, float]] = {}
    model_rows: list[dict[str, object]] = []
    all6_frac = 0.0
    extra_m = dict(extra_model_meta) if extra_model_meta else {}
    extra_c = dict(extra_contrast_meta) if extra_contrast_meta else {}

    for model_id in list(sel_df["model_id"].drop_duplicates()):
        yp_str = pivot_pred[model_id].astype(str).to_numpy()
        yp_idx = np.asarray([cat_to_idx[c] for c in yp_str], dtype=np.int64)
        point = classification_metrics(yt_str, yp_str)
        b_f1, b_ba, b_acc, n_pres = _fast_six_class_bootstrap_metrics(yt_idx, yp_idx, boot_indices)
        all6_frac = float(np.mean(n_pres == len(EXPECTED_CATEGORIES)))
        boot_cache[model_id] = (b_f1, b_ba, b_acc)
        point_cache[model_id] = point

        meta_row = sel_df.loc[sel_df["model_id"].eq(model_id)].iloc[0]
        row_m: dict[str, object] = {
            "model_id": model_id,
            "method": str(meta_row["method"]),
            "feature_set": str(meta_row["feature_set"]),
            "n_val_images": int(len(meta)),
            "n_val_groups": int(len(set(groups))),
            "n_resamples": int(n_resamples),
            "bootstrap_seed": int(seed),
            "all_6_categories_present_frac": all6_frac,
            "mean_present_categories": float(np.mean(n_pres)),
            "val_macro_f1": float(point["macro_f1"]),
            "val_macro_f1_ci95_low": float(np.percentile(b_f1, 2.5)),
            "val_macro_f1_ci95_high": float(np.percentile(b_f1, 97.5)),
            "val_balanced_accuracy": float(point["balanced_accuracy"]),
            "val_bal_acc_ci95_low": float(np.percentile(b_ba, 2.5)),
            "val_bal_acc_ci95_high": float(np.percentile(b_ba, 97.5)),
            "val_accuracy": float(point["accuracy"]),
            "val_acc_ci95_low": float(np.percentile(b_acc, 2.5)),
            "val_acc_ci95_high": float(np.percentile(b_acc, 97.5)),
        }
        row_m.update(extra_m)
        model_rows.append(row_m)

    contrast_rows: list[dict[str, object]] = []

    def _append_contrast(
        *,
        contrast_group: str,
        contrast_id: str,
        label_a: str,
        label_b: str,
        description: str,
        pt_a: Mapping[str, float],
        pt_b: Mapping[str, float],
        arr_a: tuple[np.ndarray, np.ndarray, np.ndarray],
        arr_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        f1_a, ba_a, acc_a = arr_a
        f1_b, ba_b, acc_b = arr_b
        d_f1 = f1_a - f1_b
        d_ba = ba_a - ba_b
        d_acc = acc_a - acc_b
        row_c: dict[str, object] = {
            "contrast_group": contrast_group,
            "contrast_id": contrast_id,
            "model_a": label_a,
            "model_b": label_b,
            "description": description,
            "val_macro_f1_a": float(pt_a["macro_f1"]),
            "val_macro_f1_b": float(pt_b["macro_f1"]),
            "delta_val_macro_f1": float(pt_a["macro_f1"] - pt_b["macro_f1"]),
            "delta_val_macro_f1_ci95_low": float(np.percentile(d_f1, 2.5)),
            "delta_val_macro_f1_ci95_high": float(np.percentile(d_f1, 97.5)),
            prop_gt_zero_col: float(np.mean(d_f1 > 0.0)),
            "delta_val_balanced_accuracy": float(pt_a["balanced_accuracy"] - pt_b["balanced_accuracy"]),
            "delta_val_bal_acc_ci95_low": float(np.percentile(d_ba, 2.5)),
            "delta_val_bal_acc_ci95_high": float(np.percentile(d_ba, 97.5)),
            "delta_val_accuracy": float(pt_a["accuracy"] - pt_b["accuracy"]),
            "delta_val_acc_ci95_low": float(np.percentile(d_acc, 2.5)),
            "delta_val_acc_ci95_high": float(np.percentile(d_acc, 97.5)),
            "all_6_categories_present_frac": all6_frac,
        }
        row_c.update(extra_c)
        contrast_rows.append(row_c)

    for contrast_id, model_a, model_b, label in contrasts:
        if model_a not in boot_cache or model_b not in boot_cache:
            continue
        _append_contrast(
            contrast_group="strict_2x3",
            contrast_id=contrast_id,
            label_a=model_a,
            label_b=model_b,
            description=label,
            pt_a=point_cache[model_a],
            pt_b=point_cache[model_b],
            arr_a=boot_cache[model_a],
            arr_b=boot_cache[model_b],
        )

    if v2_pivot is not None:
        for model_id in list(sel_df["model_id"].drop_duplicates()):
            if model_id not in v2_pivot.columns:
                continue
            yp_v2_str = v2_pivot[model_id].astype(str).to_numpy()
            yp_v2_idx = np.asarray([cat_to_idx[c] for c in yp_v2_str], dtype=np.int64)
            pt_v2 = classification_metrics(yt_str, yp_v2_str)
            v2_f1, v2_ba, v2_acc, _ = _fast_six_class_bootstrap_metrics(yt_idx, yp_v2_idx, boot_indices)
            _append_contrast(
                contrast_group="strict_v3_minus_fixed_v2",
                contrast_id=f"v3_minus_v2_{model_id}",
                label_a=f"{model_id} (v3 strict)",
                label_b=f"{model_id} (v2 fixed)",
                description=f"Strict fold-aware CV minus fixed-upstream CV for {model_id}",
                pt_a=point_cache[model_id],
                pt_b=pt_v2,
                arr_a=boot_cache[model_id],
                arr_b=(v2_f1, v2_ba, v2_acc),
            )

    return pd.DataFrame(model_rows), pd.DataFrame(contrast_rows)


def compute_val_bootstrap_models_and_contrasts(
    val_predictions_df: pd.DataFrame,
    contrasts: Sequence[tuple[str, str, str, str]] = STRICT_KEY_CONTRASTS,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    v2_predictions_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute paired group-blocked bootstrap CIs on validation groups for models and contrasts."""
    sel_df, pivot_pred, meta, yt_str, yt_idx, groups, cat_to_idx, v2_pivot = _prepare_val_bootstrap_inputs(
        val_predictions_df, v2_predictions_df=v2_predictions_df
    )
    boot_indices = _group_bootstrap_indices(groups, n_resamples=n_resamples, seed=seed)
    return _paired_bootstrap_tables_from_indices(
        sel_df=sel_df,
        pivot_pred=pivot_pred,
        meta=meta,
        yt_str=yt_str,
        yt_idx=yt_idx,
        groups=groups,
        cat_to_idx=cat_to_idx,
        boot_indices=boot_indices,
        contrasts=contrasts,
        seed=seed,
        v2_pivot=v2_pivot,
        prop_gt_zero_col="prob_delta_macro_f1_gt_0",
    )


def _conditional_all_class_group_bootstrap_indices(
    groups: Sequence[str] | np.ndarray,
    cat_indices: np.ndarray,
    *,
    n_classes: int = len(EXPECTED_CATEGORIES),
    n_accepted: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    max_draw_multiplier: int = 50,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Draw deterministic group-blocked bootstrap replicates conditional on all classes present."""
    g_arr = np.asarray(groups, dtype=object)
    unique_groups = np.array(sorted(set(g_arr)), dtype=object)
    n_groups = len(unique_groups)
    group_to_rows = [np.flatnonzero(g_arr == g) for g in unique_groups]
    group_cat_mask = np.zeros((n_groups, n_classes), dtype=bool)
    for g_idx, rows in enumerate(group_to_rows):
        present = np.unique(cat_indices[rows])
        group_cat_mask[g_idx, present] = True

    if not np.all(np.any(group_cat_mask, axis=0)):
        raise ValueError("Source cohort does not contain all expected classes across groups.")

    rng = np.random.default_rng(seed)
    accepted_indices: list[np.ndarray] = []
    total_draws = 0
    max_draws = int(n_accepted) * int(max_draw_multiplier)

    while len(accepted_indices) < int(n_accepted):
        if total_draws >= max_draws:
            raise RuntimeError(
                f"Reached max_draws={max_draws} with only {len(accepted_indices)}/{n_accepted} "
                "all-class group bootstrap replicates."
            )
        sampled_group_pos = rng.choice(n_groups, size=n_groups, replace=True)
        total_draws += 1
        if np.all(np.any(group_cat_mask[sampled_group_pos], axis=0)):
            rows = np.concatenate([group_to_rows[int(idx)] for idx in sampled_group_pos])
            accepted_indices.append(rows)

    rejected_draws = total_draws - len(accepted_indices)
    stats: dict[str, object] = {
        "bootstrap_scheme": "conditional_all6_group_blocked_paired_bootstrap",
        "n_groups": int(n_groups),
        "n_classes": int(n_classes),
        "n_accepted": int(len(accepted_indices)),
        "total_draws": int(total_draws),
        "rejected_draws": int(rejected_draws),
        "acceptance_rate": float(len(accepted_indices) / total_draws),
        "seed": int(seed),
    }
    return accepted_indices, stats


def compute_conditional_all6_val_bootstrap(
    val_predictions_df: pd.DataFrame,
    unconditional_models_df: pd.DataFrame | None = None,
    unconditional_contrasts_df: pd.DataFrame | None = None,
    *,
    contrasts: Sequence[tuple[str, str, str, str]] = STRICT_KEY_CONTRASTS,
    n_accepted: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    v2_predictions_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compute all-6-class conditional group-blocked bootstrap CIs and compare vs unconditional draws."""
    sel_df, pivot_pred, meta, yt_str, yt_idx, groups, cat_to_idx, v2_pivot = _prepare_val_bootstrap_inputs(
        val_predictions_df, v2_predictions_df=v2_predictions_df
    )
    if unconditional_models_df is None or unconditional_contrasts_df is None:
        uncond_boot = _group_bootstrap_indices(groups, n_resamples=n_accepted, seed=seed)
        uncond_m_df, uncond_c_df = _paired_bootstrap_tables_from_indices(
            sel_df=sel_df,
            pivot_pred=pivot_pred,
            meta=meta,
            yt_str=yt_str,
            yt_idx=yt_idx,
            groups=groups,
            cat_to_idx=cat_to_idx,
            boot_indices=uncond_boot,
            contrasts=contrasts,
            seed=seed,
            v2_pivot=v2_pivot,
            prop_gt_zero_col="prob_delta_macro_f1_gt_0",
        )
    else:
        uncond_m_df = unconditional_models_df.copy()
        uncond_c_df = unconditional_contrasts_df.copy()

    cond_indices, draw_stats = _conditional_all_class_group_bootstrap_indices(
        groups,
        yt_idx,
        n_classes=len(EXPECTED_CATEGORIES),
        n_accepted=n_accepted,
        seed=seed,
    )
    cond_models_df, cond_contrasts_df = _paired_bootstrap_tables_from_indices(
        sel_df=sel_df,
        pivot_pred=pivot_pred,
        meta=meta,
        yt_str=yt_str,
        yt_idx=yt_idx,
        groups=groups,
        cat_to_idx=cat_to_idx,
        boot_indices=cond_indices,
        contrasts=contrasts,
        seed=seed,
        v2_pivot=v2_pivot,
        prop_gt_zero_col="prop_bootstrap_delta_gt_0",
        extra_model_meta={
            "bootstrap_scheme": draw_stats["bootstrap_scheme"],
            "total_draws": draw_stats["total_draws"],
            "rejected_draws": draw_stats["rejected_draws"],
            "acceptance_rate": draw_stats["acceptance_rate"],
        },
        extra_contrast_meta={
            "bootstrap_scheme": draw_stats["bootstrap_scheme"],
            "total_draws": draw_stats["total_draws"],
            "rejected_draws": draw_stats["rejected_draws"],
            "acceptance_rate": draw_stats["acceptance_rate"],
        },
    )

    uncond_prop_col = (
        "prob_delta_macro_f1_gt_0" if "prob_delta_macro_f1_gt_0" in uncond_c_df.columns else "prop_bootstrap_delta_gt_0"
    )
    comp_rows: list[dict[str, object]] = []
    uncond_m_map = uncond_m_df.set_index("model_id")
    for _, cm in cond_models_df.iterrows():
        mid = str(cm["model_id"])
        um = uncond_m_map.loc[mid]
        u_w = float(um["val_macro_f1_ci95_high"] - um["val_macro_f1_ci95_low"])
        c_w = float(cm["val_macro_f1_ci95_high"] - cm["val_macro_f1_ci95_low"])
        comp_rows.append(
            {
                "record_type": "model",
                "item_id": mid,
                "description": f"{cm['method']} / Set {cm['feature_set']}",
                "point_macro_f1": float(cm["val_macro_f1"]),
                "uncond_macro_f1_ci95_low": float(um["val_macro_f1_ci95_low"]),
                "uncond_macro_f1_ci95_high": float(um["val_macro_f1_ci95_high"]),
                "uncond_macro_f1_ci95_width": u_w,
                "cond_macro_f1_ci95_low": float(cm["val_macro_f1_ci95_low"]),
                "cond_macro_f1_ci95_high": float(cm["val_macro_f1_ci95_high"]),
                "cond_macro_f1_ci95_width": c_w,
                "delta_macro_f1_ci95_low": float(cm["val_macro_f1_ci95_low"] - um["val_macro_f1_ci95_low"]),
                "delta_macro_f1_ci95_high": float(cm["val_macro_f1_ci95_high"] - um["val_macro_f1_ci95_high"]),
                "delta_macro_f1_ci95_width": c_w - u_w,
                "width_ratio_macro_f1": (c_w / u_w) if u_w > 0.0 else 1.0,
                "point_bal_acc": float(cm["val_balanced_accuracy"]),
                "uncond_bal_acc_ci95_low": float(um["val_bal_acc_ci95_low"]),
                "uncond_bal_acc_ci95_high": float(um["val_bal_acc_ci95_high"]),
                "cond_bal_acc_ci95_low": float(cm["val_bal_acc_ci95_low"]),
                "cond_bal_acc_ci95_high": float(cm["val_bal_acc_ci95_high"]),
                "point_accuracy": float(cm["val_accuracy"]),
                "uncond_acc_ci95_low": float(um["val_acc_ci95_low"]),
                "uncond_acc_ci95_high": float(um["val_acc_ci95_high"]),
                "cond_acc_ci95_low": float(cm["val_acc_ci95_low"]),
                "cond_acc_ci95_high": float(cm["val_acc_ci95_high"]),
                "uncond_prop_gt_0": np.nan,
                "cond_prop_gt_0": np.nan,
                "uncond_excludes_zero": np.nan,
                "cond_excludes_zero": np.nan,
                "ci_zero_conclusion_unchanged": np.nan,
            }
        )

    uncond_c_map = uncond_c_df.set_index("contrast_id")
    changed_f1: list[str] = []
    changed_ba: list[str] = []
    changed_acc: list[str] = []
    for _, cc in cond_contrasts_df.loc[cond_contrasts_df["contrast_group"].eq("strict_2x3")].iterrows():
        cid = str(cc["contrast_id"])
        uc = uncond_c_map.loc[cid]
        u_w = float(uc["delta_val_macro_f1_ci95_high"] - uc["delta_val_macro_f1_ci95_low"])
        c_w = float(cc["delta_val_macro_f1_ci95_high"] - cc["delta_val_macro_f1_ci95_low"])
        u_excl = bool(
            float(uc["delta_val_macro_f1_ci95_low"]) > 0.0
            or float(uc["delta_val_macro_f1_ci95_high"]) < 0.0
        )
        c_excl = bool(
            float(cc["delta_val_macro_f1_ci95_low"]) > 0.0
            or float(cc["delta_val_macro_f1_ci95_high"]) < 0.0
        )
        if u_excl != c_excl:
            changed_f1.append(cid)
        u_ba_excl = bool(
            float(uc["delta_val_bal_acc_ci95_low"]) > 0.0 or float(uc["delta_val_bal_acc_ci95_high"]) < 0.0
        )
        c_ba_excl = bool(
            float(cc["delta_val_bal_acc_ci95_low"]) > 0.0 or float(cc["delta_val_bal_acc_ci95_high"]) < 0.0
        )
        if u_ba_excl != c_ba_excl:
            changed_ba.append(cid)
        u_ac_excl = bool(
            float(uc["delta_val_acc_ci95_low"]) > 0.0 or float(uc["delta_val_acc_ci95_high"]) < 0.0
        )
        c_ac_excl = bool(
            float(cc["delta_val_acc_ci95_low"]) > 0.0 or float(cc["delta_val_acc_ci95_high"]) < 0.0
        )
        if u_ac_excl != c_ac_excl:
            changed_acc.append(cid)

        comp_rows.append(
            {
                "record_type": "contrast",
                "item_id": cid,
                "description": str(cc["description"]),
                "point_macro_f1": float(cc["delta_val_macro_f1"]),
                "uncond_macro_f1_ci95_low": float(uc["delta_val_macro_f1_ci95_low"]),
                "uncond_macro_f1_ci95_high": float(uc["delta_val_macro_f1_ci95_high"]),
                "uncond_macro_f1_ci95_width": u_w,
                "cond_macro_f1_ci95_low": float(cc["delta_val_macro_f1_ci95_low"]),
                "cond_macro_f1_ci95_high": float(cc["delta_val_macro_f1_ci95_high"]),
                "cond_macro_f1_ci95_width": c_w,
                "delta_macro_f1_ci95_low": float(
                    cc["delta_val_macro_f1_ci95_low"] - uc["delta_val_macro_f1_ci95_low"]
                ),
                "delta_macro_f1_ci95_high": float(
                    cc["delta_val_macro_f1_ci95_high"] - uc["delta_val_macro_f1_ci95_high"]
                ),
                "delta_macro_f1_ci95_width": c_w - u_w,
                "width_ratio_macro_f1": (c_w / u_w) if u_w > 0.0 else 1.0,
                "point_bal_acc": float(cc["delta_val_balanced_accuracy"]),
                "uncond_bal_acc_ci95_low": float(uc["delta_val_bal_acc_ci95_low"]),
                "uncond_bal_acc_ci95_high": float(uc["delta_val_bal_acc_ci95_high"]),
                "cond_bal_acc_ci95_low": float(cc["delta_val_bal_acc_ci95_low"]),
                "cond_bal_acc_ci95_high": float(cc["delta_val_bal_acc_ci95_high"]),
                "point_accuracy": float(cc["delta_val_accuracy"]),
                "uncond_acc_ci95_low": float(uc["delta_val_acc_ci95_low"]),
                "uncond_acc_ci95_high": float(uc["delta_val_acc_ci95_high"]),
                "cond_acc_ci95_low": float(cc["delta_val_acc_ci95_low"]),
                "cond_acc_ci95_high": float(cc["delta_val_acc_ci95_high"]),
                "uncond_prop_gt_0": float(uc[uncond_prop_col]),
                "cond_prop_gt_0": float(cc["prop_bootstrap_delta_gt_0"]),
                "uncond_excludes_zero": u_excl,
                "cond_excludes_zero": c_excl,
                "ci_zero_conclusion_unchanged": bool(u_excl == c_excl),
            }
        )

    comparison_df = pd.DataFrame(comp_rows)
    model_ratios = comparison_df.loc[comparison_df["record_type"].eq("model"), "width_ratio_macro_f1"].astype(float)
    contrast_ratios = comparison_df.loc[comparison_df["record_type"].eq("contrast"), "width_ratio_macro_f1"].astype(
        float
    )
    summary_meta: dict[str, object] = {
        "schema": "ebhi_dip.strict_cv_bootstrap_sensitivity.v1",
        **draw_stats,
        "unconditional_all_6_categories_present_frac": float(uncond_m_df["all_6_categories_present_frac"].iloc[0]),
        "unconditional_bootstrap_seed": int(uncond_m_df["bootstrap_seed"].iloc[0]),
        "unconditional_n_resamples": int(uncond_m_df["n_resamples"].iloc[0]),
        "model_macro_f1_width_ratio_min": float(model_ratios.min()),
        "model_macro_f1_width_ratio_median": float(model_ratios.median()),
        "model_macro_f1_width_ratio_max": float(model_ratios.max()),
        "contrast_macro_f1_width_ratio_min": float(contrast_ratios.min()),
        "contrast_macro_f1_width_ratio_median": float(contrast_ratios.median()),
        "contrast_macro_f1_width_ratio_max": float(contrast_ratios.max()),
        "contrast_zero_exclusion_changed_macro_f1": changed_f1,
        "contrast_zero_exclusion_changed_balanced_accuracy": changed_ba,
        "contrast_zero_exclusion_changed_accuracy": changed_acc,
        "substantive_macro_f1_conclusions_unchanged": len(changed_f1) == 0,
        "scientific_interpretation_note": (
            "Conditional group-blocked resampling conditions on all 6 categories appearing among the 53 sampled "
            "validation groups (rejecting draws lacking Serrated adenoma [2 groups] or Normal [3 groups]). "
            "Conditional intervals ensure a fixed 6-class Macro-F1 / Balanced Accuracy definition across all 2,000 "
            "replicates, whereas unconditional intervals reflect natural finite-sample group dropout (~17.7% of draws). "
            "Empirical bootstrap proportions (prop_bootstrap_delta_gt_0) summarize resampling variability and are "
            "neither Bayesian posterior probabilities nor confirmatory hypothesis tests."
        ),
    }
    return cond_models_df, cond_contrasts_df, comparison_df, summary_meta


def _assemble_fold_matrices(
    appearance_matrix: np.ndarray,
    fold_morph: np.ndarray,
    fold_seg_app: np.ndarray,
    methods: tuple[str, ...],
    method: str,
    feature_set: str,
    primary_method: str,
) -> list[np.ndarray]:
    """Assemble (5 folds) x (n_train, d) feature matrices for one (method, feature_set)."""
    n_splits = fold_morph.shape[0]
    m_to_idx = {m: idx for idx, m in enumerate(methods)}
    if method == SHARED_APPEARANCE_METHOD and feature_set == "A":
        return [appearance_matrix for _ in range(n_splits)]
    if feature_set == "B":
        pos = m_to_idx[method]
        return [fold_morph[k, :, pos, :] for k in range(n_splits)]
    if feature_set == "C":
        pos = m_to_idx[method]
        return [np.concatenate([appearance_matrix, fold_morph[k, :, pos, :]], axis=1) for k in range(n_splits)]
    if feature_set == "A_seg":
        return [fold_seg_app[k, :, :] for k in range(n_splits)]
    if feature_set == "C_seg":
        pos = m_to_idx[primary_method]
        return [np.concatenate([fold_seg_app[k, :, :], fold_morph[k, :, pos, :]], axis=1) for k in range(n_splits)]
    raise ValueError(f"Unsupported combination: method={method!r}, feature_set={feature_set!r}")


def _run_grid_task(
    task: tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[np.ndarray], tuple[str, ...], str, str, str],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    """Worker entrypoint for evaluating one strict CV grid specification."""
    y_train, fold_splits, fold_mats, cols, m_name, fset, track = task
    return evaluate_strict_cv_grid(
        y_train=y_train,
        fold_splits=fold_splits,
        fold_feature_matrices=fold_mats,
        feature_names=cols,
        method=m_name,
        feature_set=fset,
        track=track,
        selection_key=cv_selection_key_for_svm,
    )


def run_svm_2x3_strict_experiments(
    project_root: str | Path,
    methods: Sequence[str] = COMPARISON_2X3_METHODS,
    primary_method: str = PRIMARY_FEATURE_METHOD,
    data_root_override: str | Path | None = None,
    workers: int = 1,
    use_cache: bool = True,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Svm2x3StrictExperimentOutputs:
    """Run strict fold-aware 5-fold CV and validation-blind SVM selection for 2x3 + Set A + A_seg/C_seg."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    split_df = load_split_manifest(root, verify=False)
    method_tuple = tuple(methods)

    train_meta_df = (
        split_df.loc[split_df["split"].eq("train")]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(train_meta_df) != 1535 or int(train_meta_df["group_id"].nunique()) != 247:
        raise ValueError("Unexpected train split counts in split_manifest.csv.")

    train_app, val_app, app_sha = load_shared_appearance_frame(root)
    if app_sha != split_sha:
        raise ValueError("Frozen split checksum mismatch in shared appearance table.")
    if not train_meta_df["sample_id"].astype(str).equals(train_app["sample_id"].astype(str)):
        raise ValueError("Train metadata alignment mismatch vs appearance.csv.")

    # Load canonical full-train feature tables (used ONLY post-selection for complete-train fit & val eval)
    canonical_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for m_name in method_tuple:
        tr_df, va_df, f_sha = load_classification_feature_frame(root, method=m_name, segmented=False)
        if f_sha != split_sha:
            raise ValueError(f"Frozen split checksum mismatch in {m_name} feature table.")
        canonical_frames[m_name] = (tr_df, va_df)
    tr_seg, va_seg, seg_sha = load_classification_feature_frame(root, method=primary_method, segmented=True)
    if seg_sha != split_sha:
        raise ValueError("Frozen split checksum mismatch in segmented feature table.")

    otsu_cfg = load_otsu_config(root)
    global_otsu_polarity = str(otsu_cfg["foreground_polarity"])
    global_k4_model = load_kmeans_model(root, k=4, verify_provenance=True)

    y_train = train_meta_df["category"].to_numpy(dtype=object)
    g_train = train_meta_df["group_id"].to_numpy(dtype=object)
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS)
    fold_splits = [(f_tr, f_va) for f_tr, f_va in sgkf.split(np.zeros(len(train_meta_df)), y_train, g_train)]

    data_root_path = Path(data_root_override).expanduser().resolve() if data_root_override else None
    fold_morph, fold_seg_app, fold_audit = _build_fold_feature_cache(
        project_root=root,
        train_meta_df=train_meta_df,
        fold_splits=fold_splits,
        methods=method_tuple,
        global_otsu_polarity=global_otsu_polarity,
        global_k4_model=global_k4_model,
        split_sha=split_sha,
        data_root_override=data_root_path,
        workers=workers,
        use_cache=use_cache,
    )

    # Verify independent Otsu parity whenever fold Otsu polarity matches canonical full-train polarity
    otsu_parity_max_diff: dict[str, float] = {}
    morph_cols = list(MORPHOLOGY_FEATURE_NAMES)
    for m_idx, m_name in enumerate(method_tuple):
        if m_name.startswith("otsu_"):
            diffs: list[float] = []
            can_tr = canonical_frames[m_name][0][morph_cols].to_numpy(dtype=np.float64)
            for f_idx, fa in enumerate(fold_audit):
                if fa["otsu_matches_full_train_polarity"]:
                    diffs.append(float(np.max(np.abs(fold_morph[f_idx, :, m_idx, :] - can_tr))))
            if diffs:
                max_d = max(diffs)
                otsu_parity_max_diff[m_name] = max_d
                if max_d > 1e-12:
                    raise RuntimeError(f"Otsu strict worker morphology parity drift for {m_name}: {max_d:.3e}")

    appearance_matrix = train_app[list(APPEARANCE_FEATURE_NAMES)].to_numpy(dtype=np.float64)

    # Experiment plan: Shared Set A, 6x{B, C}, and k4_M0 {A_seg, C_seg}
    experiment_specs: list[tuple[str, str, str, pd.DataFrame, pd.DataFrame, str]] = [
        ("shared_baseline", SHARED_APPEARANCE_METHOD, "A", train_app, val_app, "svm_shared_A")
    ]
    for m_name in method_tuple:
        tr_df, va_df = canonical_frames[m_name]
        for fset in ("B", "C"):
            experiment_specs.append(("factorial_2x3", m_name, fset, tr_df, va_df, f"svm_{m_name}_{fset}"))
    for fset in SENSITIVITY_FEATURE_SETS:
        experiment_specs.append(("sensitivity_seg", primary_method, fset, tr_seg, va_seg, f"svm_{primary_method}_{fset}"))

    grid_tasks = [
        (
            y_train,
            fold_splits,
            _assemble_fold_matrices(
                appearance_matrix=appearance_matrix,
                fold_morph=fold_morph,
                fold_seg_app=fold_seg_app,
                methods=method_tuple,
                method=m_name,
                feature_set=fset,
                primary_method=primary_method,
            ),
            feature_columns_for_set(fset),
            m_name,
            fset,
            track,
        )
        for track, m_name, fset, _full_tr, _full_va, _mid in experiment_specs
    ]
    if int(workers) > 1 and len(grid_tasks) > 1:
        pool_workers = min(int(workers), len(grid_tasks))
        with ProcessPoolExecutor(max_workers=pool_workers) as executor:
            grid_outputs = list(executor.map(_run_grid_task, grid_tasks))
    else:
        grid_outputs = [_run_grid_task(task) for task in grid_tasks]

    grid_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    per_cat_frames: list[pd.DataFrame] = []
    val_pred_frames: list[pd.DataFrame] = []
    model_artifacts: dict[str, dict[str, object]] = {}

    for (track, m_name, fset, full_tr_df, full_va_df, model_id), (grid_df, best_row, default_row) in zip(
        experiment_specs, grid_outputs
    ):
        cols = feature_columns_for_set(fset)
        grid_frames.append(grid_df)

        # Post-selection ONLY: fit single selected spec (+ default reference) on complete train, evaluate ONCE on val
        X_tr_raw = full_tr_df[list(cols)].to_numpy(dtype=np.float64)
        X_va_raw = full_va_df[list(cols)].to_numpy(dtype=np.float64)
        y_tr = full_tr_df["category"].to_numpy(dtype=object)
        y_va = full_va_df["category"].to_numpy(dtype=object)

        scaler_full = fit_standard_scaler(X_tr_raw, cols)
        X_tr_sc = apply_standard_scaler(X_tr_raw, scaler_full)

        if m_name == SHARED_APPEARANCE_METHOD:
            family, stage = "appearance", "none"
        else:
            family, stage = m_name.split("_", 1)

        for variant, spec_row in (("cv_selected", best_row), ("default_rbf", default_row)):
            gamma_val = _parse_gamma(str(spec_row["gamma"]))
            clf = fit_svm_model(
                X_tr_sc,
                y_tr,
                kernel=str(spec_row["kernel"]),
                C=float(spec_row["C"]),
                gamma=gamma_val,
                class_weight=str(spec_row["class_weight"]),
            )
            tr_pred = clf.predict(X_tr_sc)
            tr_scores = classification_metrics(y_tr, tr_pred)

            X_va_sc = apply_standard_scaler(X_va_raw, scaler_full)
            va_pred = clf.predict(X_va_sc)
            va_scores = classification_metrics(y_va, va_pred)

            cv_meta = {
                "n_splits": float(CV_FOLDS),
                "cv_macro_f1": float(spec_row["cv_macro_f1"]),
                "cv_fold_macro_f1_mean": float(spec_row["cv_fold_macro_f1_mean"]),
                "cv_fold_macro_f1_std": float(spec_row["cv_fold_macro_f1_std"]),
                "cv_fold_macro_f1_min": float(spec_row["cv_fold_macro_f1_min"]),
                "cv_fold_macro_f1_max": float(spec_row["cv_fold_macro_f1_max"]),
                "cv_balanced_accuracy": float(spec_row["cv_balanced_accuracy"]),
                "cv_fold_balanced_accuracy_mean": float(spec_row["cv_fold_balanced_accuracy_mean"]),
                "cv_accuracy": float(spec_row["cv_accuracy"]),
                "cv_weighted_f1": float(spec_row["cv_weighted_f1"]),
            }

            if variant == "cv_selected":
                artifact = export_svm_artifact(
                    model_id=model_id,
                    segmentation_method=m_name,
                    feature_set=fset,
                    scaler_payload=scaler_full,
                    clf=clf,
                    split_sha256=split_sha,
                    validation_metrics=va_scores,
                    cv_metrics=cv_meta,
                )
                artifact["cv_protocol"] = "5fold_stratified_group_kfold_strict_fold_aware_upstream"
                parity = verify_svm_artifact_parity(X_va_raw, clf, artifact, atol=DECISION_PARITY_ATOL)
                if not parity["exact_prediction_match"]:
                    raise RuntimeError(f"Parity mismatch for strict artifact {model_id}.")
                model_artifacts[model_id] = artifact
            else:
                temp_art = export_svm_artifact(
                    model_id=f"{model_id}__default_rbf",
                    segmentation_method=m_name,
                    feature_set=fset,
                    scaler_payload=scaler_full,
                    clf=clf,
                    split_sha256=split_sha,
                    validation_metrics=va_scores,
                    cv_metrics=cv_meta,
                )
                temp_art["cv_protocol"] = "5fold_stratified_group_kfold_strict_fold_aware_upstream"
                artifact = temp_art

            pred_df = _prediction_rows_for_model(full_va_df, X_va_raw, artifact, variant)
            pred_df["model_id"] = model_id
            val_pred_frames.append(pred_df)

            cat_df = per_category_classification_table(y_va, va_pred)
            cat_df.insert(0, "track", track)
            cat_df.insert(1, "method", m_name)
            cat_df.insert(2, "feature_set", fset)
            cat_df.insert(3, "model_variant", variant)
            cat_df.insert(4, "model_id", model_id)
            per_cat_frames.append(cat_df)

            summary_rows.append(
                {
                    "track": track,
                    "family": family,
                    "morphology_stage": stage,
                    "method": m_name,
                    "feature_set": fset,
                    "n_features": len(cols),
                    "model_variant": variant,
                    "model_id": model_id,
                    "kernel": str(spec_row["kernel"]),
                    "class_weight": str(spec_row["class_weight"]),
                    "C": float(spec_row["C"]),
                    "gamma": str(spec_row["gamma"]),
                    "train_macro_f1": tr_scores["macro_f1"],
                    "train_balanced_accuracy": tr_scores["balanced_accuracy"],
                    "train_accuracy": tr_scores["accuracy"],
                    "cv_macro_f1": float(spec_row["cv_macro_f1"]),
                    "cv_fold_macro_f1_mean": float(spec_row["cv_fold_macro_f1_mean"]),
                    "cv_fold_macro_f1_std": float(spec_row["cv_fold_macro_f1_std"]),
                    "cv_balanced_accuracy": float(spec_row["cv_balanced_accuracy"]),
                    "cv_fold_balanced_accuracy_mean": float(spec_row["cv_fold_balanced_accuracy_mean"]),
                    "cv_accuracy": float(spec_row["cv_accuracy"]),
                    "val_macro_f1": va_scores["macro_f1"],
                    "val_balanced_accuracy": va_scores["balanced_accuracy"],
                    "val_accuracy": va_scores["accuracy"],
                    "val_weighted_f1": va_scores["weighted_f1"],
                    "val_macro_precision": va_scores["macro_precision"],
                    "n_support_total": int(np.sum(clf.n_support_)),
                }
            )

    grid_results_df = pd.concat(grid_frames, ignore_index=True)
    selected_summary_df = pd.DataFrame(summary_rows)
    per_category_val_df = pd.concat(per_cat_frames, ignore_index=True)
    val_predictions_df = pd.concat(val_pred_frames, ignore_index=True)

    # Load preserved v2 outputs if present to compute exact paired protocol comparison
    v2_pred_path = root / CLASSIFICATION_2X3_DIR_REL / "svm_2x3_val_predictions.csv"
    v2_summary_path = root / CLASSIFICATION_2X3_DIR_REL / "svm_2x3_selected_summary.csv"
    v2_pred_df = pd.read_csv(v2_pred_path) if v2_pred_path.is_file() else None
    v2_sum_df = pd.read_csv(v2_summary_path) if v2_summary_path.is_file() else None

    val_boot_models_df, val_contrasts_df = compute_val_bootstrap_models_and_contrasts(
        val_predictions_df,
        contrasts=STRICT_KEY_CONTRASTS,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        v2_predictions_df=v2_pred_df,
    )
    (
        val_boot_all6_models_df,
        val_contrasts_all6_df,
        boot_sensitivity_comp_df,
        boot_sensitivity_summary,
    ) = compute_conditional_all6_val_bootstrap(
        val_predictions_df,
        unconditional_models_df=val_boot_models_df,
        unconditional_contrasts_df=val_contrasts_df,
        contrasts=STRICT_KEY_CONTRASTS,
        n_accepted=bootstrap_resamples,
        seed=bootstrap_seed,
        v2_predictions_df=v2_pred_df,
    )

    # Merge 95% val bootstrap CIs directly onto selected_summary for cv_selected rows
    boot_lookup = val_boot_models_df.set_index("model_id")
    for ci_col in (
        "val_macro_f1_ci95_low",
        "val_macro_f1_ci95_high",
        "val_bal_acc_ci95_low",
        "val_bal_acc_ci95_high",
        "val_acc_ci95_low",
        "val_acc_ci95_high",
    ):
        selected_summary_df[ci_col] = [
            float(boot_lookup.loc[row["model_id"], ci_col]) if row["model_variant"] == "cv_selected" else np.nan
            for _, row in selected_summary_df.iterrows()
        ]

    # Fixed-upstream (v2) vs strict fold-aware (v3) comparison table
    proto_rows: list[dict[str, object]] = []
    if v2_sum_df is not None:
        v2_sel = v2_sum_df.loc[v2_sum_df["model_variant"].eq("cv_selected")].set_index("model_id")
        v3_sel = selected_summary_df.loc[selected_summary_df["model_variant"].eq("cv_selected")].set_index("model_id")
        c_v2_v3 = val_contrasts_df.loc[val_contrasts_df["contrast_group"].eq("strict_v3_minus_fixed_v2")].set_index(
            "contrast_id"
        )
        for model_id, r3 in v3_sel.iterrows():
            if model_id not in v2_sel.index:
                continue
            r2 = v2_sel.loc[model_id]
            same_spec = (
                str(r2["kernel"]) == str(r3["kernel"])
                and str(r2["class_weight"]) == str(r3["class_weight"])
                and float(r2["C"]) == float(r3["C"])
                and str(r2["gamma"]) == str(r3["gamma"])
            )
            c_key = f"v3_minus_v2_{model_id}"
            ci_lo = float(c_v2_v3.loc[c_key, "delta_val_macro_f1_ci95_low"]) if c_key in c_v2_v3.index else 0.0
            ci_hi = float(c_v2_v3.loc[c_key, "delta_val_macro_f1_ci95_high"]) if c_key in c_v2_v3.index else 0.0
            proto_rows.append(
                {
                    "track": str(r3["track"]),
                    "method": str(r3["method"]),
                    "feature_set": str(r3["feature_set"]),
                    "model_id": str(model_id),
                    "fixed_v2_spec": f"{r2['kernel']} {r2['class_weight']} C={float(r2['C']):g} g={r2['gamma']}",
                    "strict_v3_spec": f"{r3['kernel']} {r3['class_weight']} C={float(r3['C']):g} g={r3['gamma']}",
                    "identical_selected_spec": bool(same_spec),
                    "fixed_v2_cv_macro_f1": float(r2["cv_macro_f1"]),
                    "strict_v3_cv_macro_f1": float(r3["cv_macro_f1"]),
                    "delta_cv_macro_f1_v3_minus_v2": float(r3["cv_macro_f1"] - r2["cv_macro_f1"]),
                    "fixed_v2_val_macro_f1": float(r2["val_macro_f1"]),
                    "strict_v3_val_macro_f1": float(r3["val_macro_f1"]),
                    "delta_val_macro_f1_v3_minus_v2": float(r3["val_macro_f1"] - r2["val_macro_f1"]),
                    "delta_val_macro_f1_ci95_low": ci_lo,
                    "delta_val_macro_f1_ci95_high": ci_hi,
                }
            )
    protocol_comp_df = pd.DataFrame(proto_rows)

    # Segmentation Dice/IoU vs strict CV & validation metrics (2x3 Set B & Set C)
    val_seg_df = load_validation_results(root)
    val_seg_sum = summarize_validation_methods(val_seg_df)
    val_seg_overall = val_seg_df.groupby("method", observed=True)[["dice", "iou"]].mean()
    control_dice = float(val_seg_sum.loc["all_foreground", "mean_category_dice"])

    shared_a_cv = selected_summary_df.loc[
        selected_summary_df["method"].eq(SHARED_APPEARANCE_METHOD)
        & selected_summary_df["model_variant"].eq("cv_selected")
    ].iloc[0]

    sel_idx = selected_summary_df.loc[selected_summary_df["track"].eq("factorial_2x3")].set_index(
        ["method", "feature_set", "model_variant"]
    )
    seg_cls_rows: list[dict[str, object]] = []
    for m_name in method_tuple:
        family, stage = m_name.split("_", 1)
        tr_df, va_df = canonical_frames[m_name]
        dev_df = pd.concat([tr_df, va_df], ignore_index=True)
        seg_row = val_seg_sum.loc[m_name]
        seg_overall_row = val_seg_overall.loc[m_name]
        b_row = sel_idx.loc[(m_name, "B", "cv_selected")]
        for fset in ("B", "C"):
            cur = sel_idx.loc[(m_name, fset, "cv_selected")]
            def_cur = sel_idx.loc[(m_name, fset, "default_rbf")]
            sub_val = b_row["val_macro_f1"] if fset == "C" else cur["val_macro_f1"]
            sub_cv = b_row["cv_macro_f1"] if fset == "C" else cur["cv_macro_f1"]
            seg_cls_rows.append(
                {
                    "family": family,
                    "morphology_stage": stage,
                    "method": m_name,
                    "feature_set": fset,
                    "n_features": int(cur["n_features"]),
                    "model_id": str(cur["model_id"]),
                    "seg_val_mean_category_dice": float(seg_row["mean_category_dice"]),
                    "seg_val_mean_category_iou": float(seg_row["mean_category_iou"]),
                    "seg_val_delta_dice_vs_control": float(seg_row["mean_category_dice"] - control_dice),
                    "seg_val_overall_dice": float(seg_overall_row["dice"]),
                    "seg_val_overall_iou": float(seg_overall_row["iou"]),
                    "train_val_mean_fg_fraction": float(dev_df["mask_foreground_fraction"].mean()),
                    "train_val_mean_component_count": float(dev_df["mask_component_count"].mean()),
                    "train_val_mean_edge_fg_fraction": float(dev_df["mask_edge_foreground_fraction"].mean()),
                    "val_mean_fg_fraction": float(va_df["mask_foreground_fraction"].mean()),
                    "val_mean_component_count": float(va_df["mask_component_count"].mean()),
                    "kernel": str(cur["kernel"]),
                    "class_weight": str(cur["class_weight"]),
                    "C": float(cur["C"]),
                    "gamma": str(cur["gamma"]),
                    "cv_macro_f1": float(cur["cv_macro_f1"]),
                    "cv_fold_macro_f1_mean": float(cur["cv_fold_macro_f1_mean"]),
                    "cv_fold_macro_f1_std": float(cur["cv_fold_macro_f1_std"]),
                    "cv_balanced_accuracy": float(cur["cv_balanced_accuracy"]),
                    "cv_accuracy": float(cur["cv_accuracy"]),
                    "val_macro_f1": float(cur["val_macro_f1"]),
                    "val_macro_f1_ci95_low": float(cur["val_macro_f1_ci95_low"]),
                    "val_macro_f1_ci95_high": float(cur["val_macro_f1_ci95_high"]),
                    "val_balanced_accuracy": float(cur["val_balanced_accuracy"]),
                    "val_accuracy": float(cur["val_accuracy"]),
                    "val_weighted_f1": float(cur["val_weighted_f1"]),
                    "default_rbf_val_macro_f1": float(def_cur["val_macro_f1"]),
                    "delta_cv_selected_vs_default_rbf_val_macro_f1": float(
                        cur["val_macro_f1"] - def_cur["val_macro_f1"]
                    ),
                    "delta_cv_macro_f1_vs_shared_A": float(cur["cv_macro_f1"] - shared_a_cv["cv_macro_f1"]),
                    "delta_val_macro_f1_vs_shared_A": float(cur["val_macro_f1"] - shared_a_cv["val_macro_f1"]),
                    "delta_val_accuracy_vs_shared_A": float(cur["val_accuracy"] - shared_a_cv["val_accuracy"]),
                    "delta_val_macro_f1_C_minus_B": float(cur["val_macro_f1"] - sub_val),
                    "delta_cv_macro_f1_C_minus_B": float(cur["cv_macro_f1"] - sub_cv),
                }
            )
    seg_vs_cls_df = pd.DataFrame(seg_cls_rows)
    correlations_descriptive = _compute_seg_vs_cls_correlations(seg_vs_cls_df)

    fold_upstream_payload: dict[str, object] = {
        "schema": "ebhi_dip.strict_cv_fold_upstream_audit.v1",
        "split_sha256": split_sha,
        "cv_folds": CV_FOLDS,
        "global_full_train_otsu_polarity": global_otsu_polarity,
        "global_full_train_k4_foreground_clusters": global_k4_model["foreground_clusters"],
        "otsu_fold_invariance_all_dark": all(f["otsu_matches_full_train_polarity"] for f in fold_audit),
        "otsu_worker_vs_canonical_max_abs_diff": otsu_parity_max_diff,
        "k4_fold_mean_center_l2_vs_full_train": [f["k4_cluster_center_mean_l2_vs_full_train"] for f in fold_audit],
        "k4_m0_fold_val_pixel_agreement_means": [f["k4_m0_val_fold_pixel_agreement_mean"] for f in fold_audit],
        "k4_m0_fold_val_iou_vs_full_train_means": [f["k4_m0_val_fold_iou_vs_full_train_mean"] for f in fold_audit],
        "folds": fold_audit,
    }

    manifest: dict[str, object] = {
        "protocol_id": "stage07_2x3_strict_fold_aware_val_blind_cv_v3",
        "study_design": "Complete 2x3 comparison (otsu/k4 x M0/M1/M2) under strict fold-aware upstream CV & val-blind hyperparameter search",
        "task": "image_level_6_class_classification",
        "split_sha256": split_sha,
        "feature_manifest_2x3_sha256": sha256_bytes(root / "results/features/feature_manifest_2x3.json"),
        "shared_appearance_sha256": sha256_bytes(root / "results/features/appearance.csv"),
        "preserved_v1_classification_manifest_sha256": sha256_bytes(
            root / "results/classification/classification_manifest.json"
        ),
        "preserved_v2_classification_manifest_sha256": (
            sha256_bytes(root / CLASSIFICATION_2X3_DIR_REL / "classification_2x3_manifest.json")
            if (root / CLASSIFICATION_2X3_DIR_REL / "classification_2x3_manifest.json").is_file()
            else None
        ),
        "cohort_counts": {
            "train_images": 1535,
            "train_groups": 247,
            "val_images": 372,
            "val_groups": 53,
            "test_images_touched_in_2x3_strict": 0,
            "test_labels_accessed_in_2x3_strict": False,
            "historical_test_note": (
                "Held-out test (319 images) was NOT touched in v3, though historical v1/Stage 08 "
                "results in results/test/ have previously evaluated test at the repository level."
            ),
        },
        "methods": list(method_tuple),
        "shared_appearance_method": SHARED_APPEARANCE_METHOD,
        "primary_method": primary_method,
        "feature_sets_tested": list(ALL_FEATURE_SETS),
        "cv_protocol": {
            "splitter": "StratifiedGroupKFold",
            "n_splits": CV_FOLDS,
            "upstream_k4_refitted_per_fold": True,
            "upstream_otsu_polarity_checked_per_fold": True,
            "scaler_fit_scope": "within_each_cv_fold_strict",
            "val_evaluated_during_grid_search": False,
            "grid_candidates_per_model": int(
                len(grid_results_df) / len(experiment_specs)
            ),
            "total_grid_rows": int(len(grid_results_df)),
            "selection_rule": "cv_selection_key_for_svm",
            "otsu_fold_invariance_all_dark": fold_upstream_payload["otsu_fold_invariance_all_dark"],
        },
        "validation_bootstrap": {
            "method": "group_blocked_paired_bootstrap_with_replacement",
            "n_val_groups": 53,
            "n_resamples": int(bootstrap_resamples),
            "seed": int(bootstrap_seed),
            "all_6_categories_present_frac": float(val_boot_models_df["all_6_categories_present_frac"].iloc[0]),
            "note": (
                "Macro-F1 & Balanced Accuracy within each bootstrap replicate average over categories "
                "with support > 0 in that replicate; Serrated adenoma occupies 2 val groups (8 images) "
                "and Normal occupies 3 val groups (10 images)."
            ),
        },
        "validation_bootstrap_conditional_all6": boot_sensitivity_summary,
        "correlations_descriptive_only_n6": {
            "warning": (
                "Computed across n=6 dependent configurations on shared validation images; "
                "treat as descriptive magnitude summaries rather than confirmatory hypothesis tests."
            ),
            "values": correlations_descriptive,
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "decision_parity_atol": DECISION_PARITY_ATOL,
        "exported_model_files": [
            (MODELS_CV_2X3_STRICT_DIR_REL / f"{model_id}.json").as_posix()
            for model_id in model_artifacts
        ],
    }

    return Svm2x3StrictExperimentOutputs(
        grid_results=grid_results_df,
        selected_summary=selected_summary_df,
        per_category_val=per_category_val_df,
        segmentation_vs_classification=seg_vs_cls_df,
        val_predictions=val_predictions_df,
        val_bootstrap_models=val_boot_models_df,
        val_paired_contrasts=val_contrasts_df,
        val_bootstrap_all6_models=val_boot_all6_models_df,
        val_paired_contrasts_all6=val_contrasts_all6_df,
        bootstrap_sensitivity_comparison=boot_sensitivity_comp_df,
        bootstrap_sensitivity_summary=boot_sensitivity_summary,
        protocol_comparison=protocol_comp_df,
        fold_upstream_audit=fold_upstream_payload,
        model_artifacts=model_artifacts,
        manifest=manifest,
        split_sha256=split_sha,
    )


def save_bootstrap_sensitivity_outputs(
    cond_models_df: pd.DataFrame,
    cond_contrasts_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    summary_meta: Mapping[str, object],
    project_root: str | Path,
) -> dict[str, Path]:
    """Write standalone conditional all-6-class bootstrap sensitivity files without touching v3 history."""
    root = Path(project_root).expanduser().resolve()
    out_dir = root / CLASSIFICATION_2X3_STRICT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    csv_map = {
        "val_bootstrap_all6_models": (out_dir / "svm_2x3_strict_val_bootstrap_all6_models.csv", cond_models_df),
        "val_paired_contrasts_all6": (out_dir / "svm_2x3_strict_val_paired_contrasts_all6.csv", cond_contrasts_df),
        "bootstrap_sensitivity_comparison": (out_dir / "svm_2x3_strict_bootstrap_comparison.csv", comparison_df),
    }
    for key, (path, df) in csv_map.items():
        df.to_csv(path, index=False, lineterminator="\n")
        written[key] = path

    summary_path = out_dir / "bootstrap_sensitivity_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(dict(summary_meta), file, indent=2, allow_nan=False)
        file.write("\n")
    written["bootstrap_sensitivity_summary"] = summary_path
    return written


def save_strict_2x3_outputs(
    outputs: Svm2x3StrictExperimentOutputs,
    project_root: str | Path,
) -> dict[str, Path]:
    """Write versioned Stage 07 strict 2x3 results and portable SVM model artifacts to disk."""
    root = Path(project_root).expanduser().resolve()
    out_dir = root / CLASSIFICATION_2X3_STRICT_DIR_REL
    model_dir = root / MODELS_CV_2X3_STRICT_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    csv_map = {
        "grid_results": (out_dir / "svm_2x3_strict_grid_results.csv", outputs.grid_results),
        "selected_summary": (out_dir / "svm_2x3_strict_selected_summary.csv", outputs.selected_summary),
        "per_category_val": (out_dir / "svm_2x3_strict_per_category_val.csv", outputs.per_category_val),
        "segmentation_vs_classification": (
            out_dir / "svm_2x3_strict_seg_vs_cls.csv",
            outputs.segmentation_vs_classification,
        ),
        "val_predictions": (out_dir / "svm_2x3_strict_val_predictions.csv", outputs.val_predictions),
        "val_bootstrap_models": (out_dir / "svm_2x3_strict_val_bootstrap_models.csv", outputs.val_bootstrap_models),
        "val_paired_contrasts": (out_dir / "svm_2x3_strict_val_paired_contrasts.csv", outputs.val_paired_contrasts),
        "protocol_comparison": (out_dir / "svm_2x3_strict_vs_fixed_upstream.csv", outputs.protocol_comparison),
    }
    for key, (path, df) in csv_map.items():
        df.to_csv(path, index=False, lineterminator="\n")
        written[key] = path

    written.update(
        save_bootstrap_sensitivity_outputs(
            outputs.val_bootstrap_all6_models,
            outputs.val_paired_contrasts_all6,
            outputs.bootstrap_sensitivity_comparison,
            outputs.bootstrap_sensitivity_summary,
            root,
        )
    )

    audit_path = out_dir / "fold_upstream_audit.json"
    with audit_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(outputs.fold_upstream_audit, file, indent=2, allow_nan=False)
        file.write("\n")
    written["fold_upstream_audit"] = audit_path

    manifest_path = out_dir / "classification_2x3_strict_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(outputs.manifest, file, indent=2, allow_nan=False)
        file.write("\n")
    written["manifest"] = manifest_path

    for model_id, artifact in outputs.model_artifacts.items():
        model_path = model_dir / f"{model_id}.json"
        with model_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(artifact, file, indent=2, allow_nan=False)
            file.write("\n")
        written[f"model:{model_id}"] = model_path

    return written
