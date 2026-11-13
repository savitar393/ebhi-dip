"""Controlled morphology ablation experiments on EBHI-DIP (Task 2)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import sklearn
import skimage
from sklearn.model_selection import StratifiedGroupKFold

from ebhi_dip.classification import (
    CV_FOLDS,
    DEFAULT_BASELINE_SPEC,
    SHARED_APPEARANCE_METHOD,
    apply_standard_scaler,
    classification_metrics,
    cv_selection_key_for_svm,
    feature_columns_for_set,
    fit_standard_scaler,
    fit_svm_model,
    load_shared_appearance_frame,
    per_category_classification_table,
)
from ebhi_dip.dataset import (
    EXPECTED_CATEGORIES,
    load_reference_mask,
    load_rgb,
    load_split_manifest,
)
from ebhi_dip.features import (
    APPEARANCE_FEATURE_NAMES,
    MORPHOLOGY_FEATURE_NAMES,
    _morphology_payload,
)
from ebhi_dip.metrics import binary_segmentation_metrics
from ebhi_dip.morphology_diagnostic import (
    AREA_BIN_EDGES,
    AREA_BIN_LABELS,
    CROSS_3X3_KERNEL,
    MIN_COMPONENT_AREA,
    NEGLIGIBLE_DELTA_THRESHOLD,
    _bin_area_percentages,
    connected_component_areas,
    render_rgb_with_gt_contour,
)
from ebhi_dip.paths import FROZEN_SPLIT_REL, FROZEN_SPLIT_SHA_REL, verify_frozen_split
from ebhi_dip.plotting import (
    FN_COLOR_RGB,
    FP_COLOR_RGB,
    TP_COLOR_RGB,
    build_error_overlay,
    configure_notebook_style,
)
from ebhi_dip.provenance import sha256_bytes
from ebhi_dip.segmentation import (
    load_kmeans_model,
    load_otsu_config,
    morphology_variants,
    nearest_clusters,
    otsu_threshold_mask,
    rgb_to_lab,
)
from ebhi_dip.strict_cv import (
    CLASSIFICATION_2X3_STRICT_DIR_REL,
    STRICT_CV_CACHE_DIR_REL,
    fit_fold_kmeans_k4_mapped,
    fit_fold_otsu_polarity,
)

# Pre-declared fixed small-hole area cutoff (< 16 px = <= 15 px, symmetric to M1 < 16 px).
MAX_SMALL_HOLE_AREA: int = 15

ABLATION_FAMILIES: tuple[str, ...] = ("otsu", "k4")
BASELINE_VARIANTS: tuple[str, ...] = ("M0", "M1", "M2")
CANDIDATE_VARIANTS: tuple[str, ...] = (
    "close_only",
    "open_only",
    "m2_area16",
    "hole_fill_small",
)
ALL_VARIANTS: tuple[str, ...] = BASELINE_VARIANTS + CANDIDATE_VARIANTS
ABLATION_FEATURE_SETS: tuple[str, ...] = ("B", "C")
RESULTS_ABLATION_DIR_REL = Path("results/morphology_ablation")
CACHE_ABLATION_FILENAME = "strict_fold_features_ablation.npz"

VARIANT_LABELS: dict[str, str] = {
    "M0": "M0 (Raw)",
    "M1": "M1 (Area ≥16)",
    "M2": "M2 (M1 + Open + Close)",
    "close_only": "CLOSE-only (M1 + Close)",
    "open_only": "OPEN-only (M1 + Open)",
    "m2_area16": "M2 + Area ≥16",
    "hole_fill_small": f"HoleFill ≤{MAX_SMALL_HOLE_AREA} px (M1)",
}

VARIANT_SHORT_LABELS: dict[str, str] = {
    "M0": "M0",
    "M1": "M1",
    "M2": "M2",
    "close_only": "CLOSE",
    "open_only": "OPEN",
    "m2_area16": "M2+A16",
    "hole_fill_small": "HoleFill",
}

VARIANT_COLORS: dict[str, str] = {
    "M0": "#4C566A",
    "M1": "#2E86AB",
    "M2": "#D95F02",
    "close_only": "#1B9E77",
    "open_only": "#E7298A",
    "m2_area16": "#7570B3",
    "hole_fill_small": "#66A61E",
}

# Predeclared compact classifier specs anchored to strict-v3 baselines (zero Val tuning).
PREDECLARED_ANCHOR_SPECS: tuple[dict[str, object], ...] = (
    {
        "spec_id": "lin_none_C3",
        "label": "Linear C=3 (none) [v3 Otsu M0-C]",
        "kernel": "linear",
        "class_weight": "none",
        "C": 3.0,
        "gamma": "scale",
    },
    {
        "spec_id": "rbf_bal_C3_g001",
        "label": "RBF C=3 γ=0.01 (bal) [v3 Otsu M1/M2-C]",
        "kernel": "rbf",
        "class_weight": "balanced",
        "C": 3.0,
        "gamma": "0.01",
    },
    {
        "spec_id": "rbf_bal_C10_g001",
        "label": "RBF C=10 γ=0.01 (bal) [v3 Otsu-B]",
        "kernel": "rbf",
        "class_weight": "balanced",
        "C": 10.0,
        "gamma": "0.01",
    },
    {
        "spec_id": "rbf_bal_C3_scale",
        "label": "RBF C=3 scale (bal) [v3 k4-C]",
        "kernel": "rbf",
        "class_weight": "balanced",
        "C": 3.0,
        "gamma": "scale",
    },
    {
        "spec_id": "default_rbf_none_C1",
        "label": "RBF C=1 scale (none) [Default]",
        "kernel": str(DEFAULT_BASELINE_SPEC["kernel"]),
        "class_weight": str(DEFAULT_BASELINE_SPEC["class_weight"]),
        "C": float(DEFAULT_BASELINE_SPEC["C"]),
        "gamma": str(DEFAULT_BASELINE_SPEC["gamma"]),
    },
)

PRIMARY_MATCHED_ANCHOR: dict[tuple[str, str], str] = {
    ("shared_A", "A"): "rbf_bal_C10_g001",
    ("otsu", "B"): "rbf_bal_C10_g001",
    ("otsu", "C"): "rbf_bal_C3_g001",
    ("k4", "B"): "rbf_bal_C3_scale",
    ("k4", "C"): "rbf_bal_C3_scale",
}


@dataclass(frozen=True)
class AblationExperimentOutputs:
    """In-memory bundle of all Task 2 segmentation, feature-shift, and strict-CV outputs."""

    manifest: dict[str, object]
    seg_train_per_image: pd.DataFrame
    seg_train_summary: pd.DataFrame
    seg_val_summary: pd.DataFrame
    area_distributions: pd.DataFrame
    feature_shifts: pd.DataFrame
    cv_results: pd.DataFrame
    fold_deltas: pd.DataFrame
    per_category_cv: pd.DataFrame
    oof_predictions: pd.DataFrame
    deterministic_cases: pd.DataFrame


def filter_small_components(
    mask: np.ndarray,
    min_area: int = MIN_COMPONENT_AREA,
    connectivity: int = 8,
) -> np.ndarray:
    """Remove connected foreground components smaller than min_area while preserving shape/dtype."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {arr.shape}.")
    if min_area < 1:
        raise ValueError(f"min_area must be >= 1, got {min_area}.")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}.")

    binary = arr.astype(bool, copy=False)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=connectivity,
    )
    if n_labels <= 1:
        return np.zeros(binary.shape, dtype=bool)

    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return np.asarray(keep[labels], dtype=bool)


def enclosed_hole_stats(
    mask: np.ndarray,
    max_hole_area: int = MAX_SMALL_HOLE_AREA,
    connectivity: int = 4,
) -> dict[str, int]:
    """Count interior background holes (non-border-touching) and small holes <= max_hole_area."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {arr.shape}.")
    if max_hole_area < 0:
        raise ValueError(f"max_hole_area must be >= 0, got {max_hole_area}.")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}.")

    binary = arr.astype(bool, copy=False)
    if not np.any(binary) or np.all(binary):
        return {"hole_count": 0, "small_hole_count": 0, "hole_pixels": 0, "small_hole_pixels": 0}

    h, w = binary.shape
    n_bg, _, bg_stats, _ = cv2.connectedComponentsWithStats(
        (~binary).astype(np.uint8),
        connectivity=connectivity,
    )
    if n_bg <= 1:
        return {"hole_count": 0, "small_hole_count": 0, "hole_pixels": 0, "small_hole_pixels": 0}

    left = bg_stats[1:, cv2.CC_STAT_LEFT]
    top = bg_stats[1:, cv2.CC_STAT_TOP]
    right = left + bg_stats[1:, cv2.CC_STAT_WIDTH]
    bottom = top + bg_stats[1:, cv2.CC_STAT_HEIGHT]
    areas = bg_stats[1:, cv2.CC_STAT_AREA]

    enclosed = (left > 0) & (top > 0) & (right < w) & (bottom < h)
    enclosed_areas = areas[enclosed]
    small_mask = enclosed_areas <= max_hole_area
    return {
        "hole_count": int(len(enclosed_areas)),
        "small_hole_count": int(np.sum(small_mask)),
        "hole_pixels": int(np.sum(enclosed_areas)),
        "small_hole_pixels": int(np.sum(enclosed_areas[small_mask])),
    }


def fill_small_enclosed_holes(
    mask: np.ndarray,
    max_hole_area: int = MAX_SMALL_HOLE_AREA,
    connectivity: int = 4,
) -> np.ndarray:
    """Fill enclosed interior background components with pixel area <= max_hole_area.

    Background components touching any image edge are exterior background and are never filled.
    4-connectivity on background is the topological dual of 8-connected foreground.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {arr.shape}.")
    if max_hole_area < 0:
        raise ValueError(f"max_hole_area must be >= 0, got {max_hole_area}.")
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}.")

    binary = arr.astype(bool, copy=True)
    if max_hole_area == 0 or not np.any(binary) or np.all(binary):
        return binary

    h, w = binary.shape
    n_bg, bg_labels, bg_stats, _ = cv2.connectedComponentsWithStats(
        (~binary).astype(np.uint8),
        connectivity=connectivity,
    )
    if n_bg <= 1:
        return binary

    left = bg_stats[1:, cv2.CC_STAT_LEFT]
    top = bg_stats[1:, cv2.CC_STAT_TOP]
    right = left + bg_stats[1:, cv2.CC_STAT_WIDTH]
    bottom = top + bg_stats[1:, cv2.CC_STAT_HEIGHT]
    areas = bg_stats[1:, cv2.CC_STAT_AREA]

    enclosed = (left > 0) & (top > 0) & (right < w) & (bottom < h)
    fill_lookup = np.zeros(n_bg, dtype=bool)
    fill_lookup[1:] = enclosed & (areas <= max_hole_area)
    return binary | fill_lookup[bg_labels]


def ablation_morphology_variants(
    mask: np.ndarray,
    min_area: int = MIN_COMPONENT_AREA,
    max_hole_area: int = MAX_SMALL_HOLE_AREA,
) -> dict[str, np.ndarray]:
    """Compute immutable baselines (M0, M1, M2) alongside the 4 controlled candidates."""
    raw = np.asarray(mask)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {raw.shape}.")

    if min_area == MIN_COMPONENT_AREA:
        base = morphology_variants(raw)
        m0, m1, m2 = base["M0"], base["M1"], base["M2"]
    else:
        m0 = raw.astype(bool, copy=True)
        m1 = filter_small_components(m0, min_area=min_area, connectivity=8)
        opened_base = (
            cv2.morphologyEx(
                m1.astype(np.uint8),
                cv2.MORPH_OPEN,
                CROSS_3X3_KERNEL,
                iterations=1,
                borderType=cv2.BORDER_REPLICATE,
            )
            > 0
        )
        m2 = (
            cv2.morphologyEx(
                opened_base.astype(np.uint8),
                cv2.MORPH_CLOSE,
                CROSS_3X3_KERNEL,
                iterations=1,
                borderType=cv2.BORDER_REPLICATE,
            )
            > 0
        )

    m1_u8 = m1.astype(np.uint8)
    close_only = (
        cv2.morphologyEx(
            m1_u8,
            cv2.MORPH_CLOSE,
            CROSS_3X3_KERNEL,
            iterations=1,
            borderType=cv2.BORDER_REPLICATE,
        )
        > 0
    )
    open_only = (
        cv2.morphologyEx(
            m1_u8,
            cv2.MORPH_OPEN,
            CROSS_3X3_KERNEL,
            iterations=1,
            borderType=cv2.BORDER_REPLICATE,
        )
        > 0
    )
    m2_area16 = filter_small_components(m2, min_area=min_area, connectivity=8)
    hole_fill_small = fill_small_enclosed_holes(
        m1, max_hole_area=max_hole_area, connectivity=4
    )

    return {
        "M0": m0,
        "M1": m1,
        "M2": m2,
        "close_only": close_only,
        "open_only": open_only,
        "m2_area16": m2_area16,
        "hole_fill_small": hole_fill_small,
    }


def _train_sample_ids_sha(sample_ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def build_strict_train_folds(
    manifest_df: pd.DataFrame,
    n_splits: int = CV_FOLDS,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Build deterministic StratifiedGroupKFold splits over train rows sorted by sample_id."""
    if manifest_df["split"].eq("test").any():
        allowed = manifest_df.loc[manifest_df["split"].isin(["train", "val"])].copy()
    else:
        allowed = manifest_df.copy()

    train_df = (
        allowed.loc[allowed["split"].eq("train")]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    cv = StratifiedGroupKFold(n_splits=n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    oof_fold_ids = np.full(len(train_df), -1, dtype=np.int32)
    for fold_idx, (tr_idx, va_idx) in enumerate(
        cv.split(train_df, train_df["category"], train_df["group_id"])
    ):
        folds.append((np.asarray(tr_idx, dtype=np.int64), np.asarray(va_idx, dtype=np.int64)))
        oof_fold_ids[va_idx] = fold_idx
        tr_groups = set(train_df.iloc[tr_idx]["group_id"])
        va_groups = set(train_df.iloc[va_idx]["group_id"])
        if tr_groups & va_groups:
            raise AssertionError(f"Group leakage detected inside fold {fold_idx}.")
    if np.any(oof_fold_ids < 0):
        raise AssertionError("Not every train sample was assigned to an OOF fold.")
    return train_df, folds, oof_fold_ids


def _resolve_fold_upstream_models(
    project_root: Path,
    train_df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load and verify fold-specific Otsu polarity and k4 cluster models without val/test access."""
    audit_path = project_root / CLASSIFICATION_2X3_STRICT_DIR_REL / "fold_upstream_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_folds = audit.get("folds", [])
        if len(audit_folds) == len(folds):
            otsu_fits: list[dict[str, object]] = []
            k4_fits: list[dict[str, object]] = []
            all_match = True
            for fold_idx, (tr_idx, _) in enumerate(folds):
                expected_sha = _train_sample_ids_sha(train_df.iloc[tr_idx]["sample_id"].tolist())
                rec = audit_folds[fold_idx]
                o_fit = rec.get("otsu_polarity_fit", {})
                k_fit = rec.get("k4_fit", {})
                if (
                    o_fit.get("train_sample_ids_sha256") != expected_sha
                    or k_fit.get("train_sample_ids_sha256") != expected_sha
                ):
                    all_match = False
                    break
                otsu_fits.append(dict(o_fit))
                k4_fits.append(dict(k_fit))
            if all_match:
                return otsu_fits, k4_fits

    otsu_fits = []
    k4_fits = []
    for tr_idx, _ in folds:
        fold_tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        otsu_fits.append(fit_fold_otsu_polarity(fold_tr_df, project_root))
        k4_fits.append(fit_fold_kmeans_k4_mapped(fold_tr_df, project_root))
    return otsu_fits, k4_fits


def _compute_seg_row(
    sample_id: str,
    category: str,
    group_id: str,
    split: str,
    family: str,
    variant: str,
    mask: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    ref: np.ndarray,
    max_hole_area: int = MAX_SMALL_HOLE_AREA,
) -> tuple[dict[str, object], np.ndarray]:
    """Compute standardized per-image segmentation metrics, pixel flips, and component areas."""
    met = binary_segmentation_metrics(mask, ref)
    met0 = binary_segmentation_metrics(m0, ref)
    met1 = binary_segmentation_metrics(m1, ref)
    areas = connected_component_areas(mask)
    h_stats = enclosed_hole_stats(mask, max_hole_area=max_hole_area)

    diff_pos_0 = mask & (~m0)
    diff_neg_0 = m0 & (~mask)
    diff_pos_1 = mask & (~m1)
    diff_neg_1 = m1 & (~mask)

    row: dict[str, object] = {
        "sample_id": sample_id,
        "category": category,
        "group_id": group_id,
        "split": split,
        "family": family,
        "variant": variant,
        "ref_fg_px": int(np.sum(ref)),
        "ref_fg_pct": 100.0 * float(met["reference_foreground_fraction"]),
        "fg_px": int(np.sum(mask)),
        "fg_pct": 100.0 * float(met["predicted_foreground_fraction"]),
        "dice": float(met["dice"]),
        "iou": float(met["iou"]),
        "precision": float(met["precision"]),
        "recall": float(met["recall"]),
        "delta_dice_vs_M0": float(met["dice"] - met0["dice"]),
        "delta_dice_vs_M1": float(met["dice"] - met1["dice"]),
        "delta_iou_vs_M0": float(met["iou"] - met0["iou"]),
        "delta_iou_vs_M1": float(met["iou"] - met1["iou"]),
        "delta_prec_vs_M0": float(met["precision"] - met0["precision"]),
        "delta_prec_vs_M1": float(met["precision"] - met1["precision"]),
        "delta_rec_vs_M0": float(met["recall"] - met0["recall"]),
        "delta_rec_vs_M1": float(met["recall"] - met1["recall"]),
        "cc_count": int(len(areas)),
        "small_cc_count": int(np.sum(areas < MIN_COMPONENT_AREA)),
        "mean_comp_area": float(np.mean(areas)) if len(areas) else 0.0,
        "median_comp_area": float(np.median(areas)) if len(areas) else 0.0,
        "hole_count": h_stats["hole_count"],
        "small_hole_count": h_stats["small_hole_count"],
        "hole_pixels": h_stats["hole_pixels"],
        "small_hole_pixels": h_stats["small_hole_pixels"],
        "removed_tp_vs_M0": int(np.sum(diff_neg_0 & ref)),
        "removed_fp_vs_M0": int(np.sum(diff_neg_0 & (~ref))),
        "added_tp_vs_M0": int(np.sum(diff_pos_0 & ref)),
        "added_fp_vs_M0": int(np.sum(diff_pos_0 & (~ref))),
        "removed_tp_vs_M1": int(np.sum(diff_neg_1 & ref)),
        "removed_fp_vs_M1": int(np.sum(diff_neg_1 & (~ref))),
        "added_tp_vs_M1": int(np.sum(diff_pos_1 & ref)),
        "added_fp_vs_M1": int(np.sum(diff_pos_1 & (~ref))),
    }
    return row, areas


def _otsu_image_worker(task: tuple[object, ...]) -> dict[str, object]:
    """Process one image for Otsu across all 7 variants and selected feature targets."""
    (
        sample_id,
        category,
        group_id,
        split,
        img_path_str,
        mask_path_str,
        polarity,
        feature_variants,
        collect_seg,
    ) = task
    rgb = load_rgb(img_path_str)
    ref = load_reference_mask(mask_path_str) if collect_seg else None
    _, m0 = otsu_threshold_mask(rgb, foreground_polarity=str(polarity))
    variants = ablation_morphology_variants(m0)

    feat_vectors: dict[str, np.ndarray] = {}
    for var in feature_variants:
        payload, _ = _morphology_payload(variants[var])
        feat_vectors[var] = np.array([payload[k] for k in MORPHOLOGY_FEATURE_NAMES], dtype=np.float64)

    seg_rows: list[dict[str, object]] = []
    area_hist_counts: dict[str, np.ndarray] = {}
    if collect_seg and ref is not None:
        m0_mask = variants["M0"]
        m1_mask = variants["M1"]
        for var in ALL_VARIANTS:
            row, areas = _compute_seg_row(
                sample_id,
                category,
                group_id,
                split,
                "otsu",
                var,
                variants[var],
                m0_mask,
                m1_mask,
                ref,
            )
            seg_rows.append(row)
            if len(areas) > 0:
                cnt, _ = np.histogram(areas, bins=np.asarray(AREA_BIN_EDGES, dtype=np.int64))
            else:
                cnt = np.zeros(len(AREA_BIN_LABELS), dtype=np.int64)
            area_hist_counts[var] = cnt.astype(np.int64)

    return {
        "sample_id": sample_id,
        "features": feat_vectors,
        "seg_rows": seg_rows,
        "area_hist": area_hist_counts,
    }


def _k4_image_worker(task: tuple[object, ...]) -> dict[str, object]:
    """Process one image for K-Means k=4 across all requested folds (reading RGB once)."""
    (
        sample_id,
        category,
        group_id,
        split,
        img_path_str,
        mask_path_str,
        fold_k4_specs,
        oof_fold_idx,
        feature_variants,
        collect_seg,
    ) = task
    rgb = load_rgb(img_path_str)
    ref = load_reference_mask(mask_path_str) if collect_seg else None
    lab_flat = rgb_to_lab(rgb).reshape(-1, 3)

    # feat_by_fold[fold_idx][variant] -> (19,)
    feat_by_fold: dict[int, dict[str, np.ndarray]] = {}
    seg_rows: list[dict[str, object]] = []
    area_hist_counts: dict[str, np.ndarray] = {}

    for fold_idx, centers, fg_clusters in fold_k4_specs:
        labels = nearest_clusters(lab_flat, centers).reshape(rgb.shape[:2])
        raw_m0 = np.isin(labels, fg_clusters)
        variants = ablation_morphology_variants(raw_m0)

        f_vecs: dict[str, np.ndarray] = {}
        for var in feature_variants:
            payload, _ = _morphology_payload(variants[var])
            f_vecs[var] = np.array(
                [payload[k] for k in MORPHOLOGY_FEATURE_NAMES], dtype=np.float64
            )
        feat_by_fold[int(fold_idx)] = f_vecs

        if collect_seg and ref is not None and int(fold_idx) == int(oof_fold_idx):
            m0_mask = variants["M0"]
            m1_mask = variants["M1"]
            for var in ALL_VARIANTS:
                row, areas = _compute_seg_row(
                    sample_id,
                    category,
                    group_id,
                    split,
                    "k4",
                    var,
                    variants[var],
                    m0_mask,
                    m1_mask,
                    ref,
                )
                seg_rows.append(row)
                if len(areas) > 0:
                    cnt, _ = np.histogram(areas, bins=np.asarray(AREA_BIN_EDGES, dtype=np.int64))
                else:
                    cnt = np.zeros(len(AREA_BIN_LABELS), dtype=np.int64)
                area_hist_counts[var] = cnt.astype(np.int64)

    return {
        "sample_id": sample_id,
        "feat_by_fold": feat_by_fold,
        "seg_rows": seg_rows,
        "area_hist": area_hist_counts,
    }


def _init_worker_process() -> None:
    """Prevent OpenCV/BLAS thread oversubscription inside multiprocessing workers."""
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _run_parallel_tasks(worker_fn, tasks: list[tuple[object, ...]], max_workers: int) -> list[dict[str, object]]:
    """Execute worker tasks deterministically in order using ProcessPoolExecutor or sequentially."""
    if max_workers <= 1 or len(tasks) < 16:
        return [worker_fn(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_process) as pool:
        return list(pool.map(worker_fn, tasks, chunksize=max(1, len(tasks) // (max_workers * 3))))


def _load_baseline_fold_cache(
    project_root: Path,
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load exact baseline M0/M1/M2 morphology feature tensors from strict_fold_cache.npz if matching."""
    cache_path = project_root / STRICT_CV_CACHE_DIR_REL / "strict_fold_cache.npz"
    if not cache_path.is_file():
        return None
    npz = np.load(cache_path, allow_pickle=True)
    cached_ids = [str(s) for s in npz["sample_ids"]]
    train_ids = train_df["sample_id"].tolist()
    if cached_ids != train_ids:
        return None

    methods = [str(m) for m in npz["methods"]]
    fold_morph = np.asarray(npz["fold_morph"], dtype=np.float64)  # (5, 1535, 6, 19)
    otsu_base = np.empty((len(train_ids), len(BASELINE_VARIANTS), len(MORPHOLOGY_FEATURE_NAMES)), dtype=np.float64)
    k4_base = np.empty((CV_FOLDS, len(train_ids), len(BASELINE_VARIANTS), len(MORPHOLOGY_FEATURE_NAMES)), dtype=np.float64)
    for v_idx, st in enumerate(BASELINE_VARIANTS):
        otsu_base[:, v_idx, :] = fold_morph[0, :, methods.index(f"otsu_{st}"), :]
        for f in range(CV_FOLDS):
            k4_base[f, :, v_idx, :] = fold_morph[f, :, methods.index(f"k4_{st}"), :]
    return otsu_base, k4_base


def _load_or_compute_train_features_and_seg(
    project_root: Path,
    train_df: pd.DataFrame,
    otsu_fits: list[dict[str, object]],
    k4_fits: list[dict[str, object]],
    oof_fold_ids: np.ndarray,
    max_workers: int,
    use_ablation_cache: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Compute or load verified strict fold features + OOF train segmentation tables for all 7 variants.

    Returns:
        otsu_morph: (5, N, 7, 19)
        k4_morph:   (5, N, 7, 19)
        seg_train_df: (N * 2 * 7 rows)
        area_dist_df: pooled component area histogram shares on train OOF
    """
    n = len(train_df)
    n_var = len(ALL_VARIANTS)
    n_feat = len(MORPHOLOGY_FEATURE_NAMES)
    cache_path = project_root / RESULTS_ABLATION_DIR_REL / CACHE_ABLATION_FILENAME
    seg_csv_path = project_root / RESULTS_ABLATION_DIR_REL / "segmentation_train_oof_per_image.csv"
    area_csv_path = project_root / RESULTS_ABLATION_DIR_REL / "component_area_distributions_train_oof.csv"

    expected_sha = _train_sample_ids_sha(train_df["sample_id"].tolist())
    if (
        use_ablation_cache
        and cache_path.is_file()
        and seg_csv_path.is_file()
        and area_csv_path.is_file()
    ):
        npz = np.load(cache_path, allow_pickle=True)
        if (
            str(npz["train_sample_ids_sha256"]) == expected_sha
            and list(npz["variants"]) == list(ALL_VARIANTS)
            and int(npz["max_small_hole_area"]) == MAX_SMALL_HOLE_AREA
        ):
            return (
                np.asarray(npz["otsu_morph"], dtype=np.float64),
                np.asarray(npz["k4_morph"], dtype=np.float64),
                pd.read_csv(seg_csv_path),
                pd.read_csv(area_csv_path),
            )

    baseline_cache = _load_baseline_fold_cache(project_root, train_df)
    feature_targets = CANDIDATE_VARIANTS if baseline_cache is not None else ALL_VARIANTS

    # Verify Otsu polarity fold invariance before deduplicating computation by polarity.
    polarities = [str(f["chosen_polarity"]) for f in otsu_fits]
    unique_pols = sorted(set(polarities))
    if baseline_cache is not None and len(unique_pols) != 1:
        feature_targets = ALL_VARIANTS
        baseline_cache = None
    otsu_morph = np.empty((CV_FOLDS, n, n_var, n_feat), dtype=np.float64)
    otsu_seg_rows: list[dict[str, object]] = []
    area_bin_totals: dict[tuple[str, str], np.ndarray] = {
        (fam, var): np.zeros(len(AREA_BIN_LABELS), dtype=np.int64)
        for fam in ABLATION_FAMILIES
        for var in ALL_VARIANTS
    }

    pol_feature_map: dict[str, np.ndarray] = {}
    for pol_idx, pol in enumerate(unique_pols):
        collect_seg = pol_idx == 0 and len(unique_pols) == 1
        tasks = [
            (
                rec.sample_id,
                rec.category,
                rec.group_id,
                "train",
                str(project_root / rec.image_path),
                str(project_root / rec.mask_path),
                pol,
                feature_targets,
                collect_seg,
            )
            for rec in train_df.itertuples(index=False)
        ]
        res = _run_parallel_tasks(_otsu_image_worker, tasks, max_workers=max_workers)
        print(f"  [ablation] Otsu polarity={pol} done ({len(res)} train imgs)", flush=True)
        mat = np.empty((n, n_var, n_feat), dtype=np.float64)
        if baseline_cache is not None:
            mat[:, : len(BASELINE_VARIANTS), :] = baseline_cache[0]
        for idx, item in enumerate(res):
            for var in feature_targets:
                mat[idx, ALL_VARIANTS.index(var), :] = item["features"][var]
            if collect_seg:
                otsu_seg_rows.extend(item["seg_rows"])
                for var, cnt in item["area_hist"].items():
                    area_bin_totals[("otsu", var)] += cnt
        pol_feature_map[pol] = mat

    for f_idx, pol in enumerate(polarities):
        otsu_morph[f_idx] = pol_feature_map[pol]

    # K-Means k=4 across all 5 folds in a single I/O pass per image.
    fold_k4_specs = [
        (
            f_idx,
            np.asarray(k4_fits[f_idx]["cluster_centers"], dtype=np.float64),
            np.asarray(k4_fits[f_idx]["foreground_clusters"], dtype=np.int64),
        )
        for f_idx in range(CV_FOLDS)
    ]
    k4_tasks = [
        (
            rec.sample_id,
            rec.category,
            rec.group_id,
            "train",
            str(project_root / rec.image_path),
            str(project_root / rec.mask_path),
            fold_k4_specs,
            int(oof_fold_ids[idx]),
            feature_targets,
            True,
        )
        for idx, rec in enumerate(train_df.itertuples(index=False))
    ]
    k4_res = _run_parallel_tasks(_k4_image_worker, k4_tasks, max_workers=max_workers)
    print(f"  [ablation] k4 5-fold features+seg done ({len(k4_res)} train imgs)", flush=True)
    k4_morph = np.empty((CV_FOLDS, n, n_var, n_feat), dtype=np.float64)
    if baseline_cache is not None:
        k4_morph[:, :, : len(BASELINE_VARIANTS), :] = baseline_cache[1]
    k4_seg_rows: list[dict[str, object]] = []
    for idx, item in enumerate(k4_res):
        for f_idx in range(CV_FOLDS):
            for var in feature_targets:
                k4_morph[f_idx, idx, ALL_VARIANTS.index(var), :] = item["feat_by_fold"][f_idx][var]
        k4_seg_rows.extend(item["seg_rows"])
        for var, cnt in item["area_hist"].items():
            area_bin_totals[("k4", var)] += cnt

    seg_train_df = (
        pd.DataFrame(otsu_seg_rows + k4_seg_rows)
        .sort_values(["family", "variant", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    area_rows: list[dict[str, object]] = []
    for fam in ABLATION_FAMILIES:
        for var in ALL_VARIANTS:
            cnts = area_bin_totals[(fam, var)]
            tot = int(np.sum(cnts))
            for b_idx, b_lbl in enumerate(AREA_BIN_LABELS):
                area_rows.append(
                    {
                        "family": fam,
                        "variant": var,
                        "area_bin": b_lbl,
                        "count": int(cnts[b_idx]),
                        "total_components": tot,
                        "share_pct": (100.0 * float(cnts[b_idx]) / tot) if tot > 0 else 0.0,
                    }
                )
    area_dist_df = pd.DataFrame(area_rows)
    return otsu_morph, k4_morph, seg_train_df, area_dist_df


def _compute_full_train_k4_and_val_readonly(
    project_root: Path,
    train_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    max_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Evaluate frozen full-train upstream artifacts on val (read-only) and train k4 (parity)."""
    val_df = (
        manifest_df.loc[manifest_df["split"].eq("val")]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    otsu_cfg = load_otsu_config(project_root)
    km_model = load_kmeans_model(project_root, k=4, verify_provenance=True)
    polarity = str(otsu_cfg["foreground_polarity"])
    km_centers = np.asarray(km_model["cluster_centers"], dtype=np.float64)
    km_fg = np.asarray(km_model["foreground_clusters"], dtype=np.int64)

    otsu_tasks = [
        (
            rec.sample_id,
            rec.category,
            rec.group_id,
            "val",
            str(project_root / rec.image_path),
            str(project_root / rec.mask_path),
            polarity,
            ALL_VARIANTS,
            True,
        )
        for rec in val_df.itertuples(index=False)
    ]
    otsu_res = _run_parallel_tasks(_otsu_image_worker, otsu_tasks, max_workers=max_workers)

    k4_val_tasks = [
        (
            rec.sample_id,
            rec.category,
            rec.group_id,
            "val",
            str(project_root / rec.image_path),
            str(project_root / rec.mask_path),
            [(0, km_centers, km_fg)],
            0,
            ALL_VARIANTS,
            True,
        )
        for rec in val_df.itertuples(index=False)
    ]
    k4_val_res = _run_parallel_tasks(_k4_image_worker, k4_val_tasks, max_workers=max_workers)

    k4_train_tasks = [
        (
            rec.sample_id,
            rec.category,
            rec.group_id,
            "train",
            str(project_root / rec.image_path),
            str(project_root / rec.mask_path),
            [(0, km_centers, km_fg)],
            0,
            ALL_VARIANTS,
            False,
        )
        for rec in train_df.itertuples(index=False)
    ]
    k4_train_res = _run_parallel_tasks(_k4_image_worker, k4_train_tasks, max_workers=max_workers)

    n_val = len(val_df)
    n_var = len(ALL_VARIANTS)
    n_feat = len(MORPHOLOGY_FEATURE_NAMES)
    val_otsu_morph = np.empty((n_val, n_var, n_feat), dtype=np.float64)
    val_k4_morph = np.empty((n_val, n_var, n_feat), dtype=np.float64)
    full_train_k4_morph = np.empty((len(train_df), n_var, n_feat), dtype=np.float64)
    rows: list[dict[str, object]] = []

    for idx, item in enumerate(otsu_res):
        for v_idx, var in enumerate(ALL_VARIANTS):
            val_otsu_morph[idx, v_idx] = item["features"][var]
        rows.extend(item["seg_rows"])

    for idx, item in enumerate(k4_val_res):
        for v_idx, var in enumerate(ALL_VARIANTS):
            val_k4_morph[idx, v_idx] = item["feat_by_fold"][0][var]
        rows.extend(item["seg_rows"])

    for idx, item in enumerate(k4_train_res):
        for v_idx, var in enumerate(ALL_VARIANTS):
            full_train_k4_morph[idx, v_idx] = item["feat_by_fold"][0][var]

    seg_val_df = (
        pd.DataFrame(rows)
        .sort_values(["family", "variant", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return val_otsu_morph, val_k4_morph, full_train_k4_morph, seg_val_df


def summarize_ablation_segmentation(seg_per_image_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize segmentation quality, coverage, components, holes, and TP/FP pixel flips."""
    metric_cols = (
        "dice",
        "iou",
        "precision",
        "recall",
        "fg_pct",
        "ref_fg_pct",
        "delta_dice_vs_M0",
        "delta_dice_vs_M1",
        "delta_iou_vs_M0",
        "delta_prec_vs_M1",
        "delta_rec_vs_M1",
        "cc_count",
        "small_cc_count",
        "mean_comp_area",
        "hole_count",
        "small_hole_count",
        "removed_tp_vs_M0",
        "removed_fp_vs_M0",
        "added_tp_vs_M0",
        "added_fp_vs_M0",
        "removed_tp_vs_M1",
        "removed_fp_vs_M1",
        "added_tp_vs_M1",
        "added_fp_vs_M1",
    )
    rows: list[dict[str, object]] = []
    for family in ABLATION_FAMILIES:
        for variant in ALL_VARIANTS:
            sub = seg_per_image_df.loc[
                seg_per_image_df["family"].eq(family)
                & seg_per_image_df["variant"].eq(variant)
            ]
            cat_means = sub.groupby("category", observed=True)[list(metric_cols)].mean()
            macro_vals = {col: float(cat_means[col].mean()) for col in metric_cols}
            overall_vals = {f"overall_{col}": float(sub[col].mean()) for col in ("dice", "iou", "precision", "recall", "fg_pct", "cc_count", "small_cc_count")}

            d_m0 = sub["delta_dice_vs_M0"].to_numpy(dtype=np.float64)
            d_m1 = sub["delta_dice_vs_M1"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "n_images": int(len(sub)),
                    "macro_dice": macro_vals["dice"],
                    "macro_iou": macro_vals["iou"],
                    "macro_precision": macro_vals["precision"],
                    "macro_recall": macro_vals["recall"],
                    "macro_fg_pct": macro_vals["fg_pct"],
                    "macro_delta_dice_vs_M0": macro_vals["delta_dice_vs_M0"],
                    "macro_delta_dice_vs_M1": macro_vals["delta_dice_vs_M1"],
                    "macro_delta_iou_vs_M0": macro_vals["delta_iou_vs_M0"],
                    "macro_delta_prec_vs_M1": macro_vals["delta_prec_vs_M1"],
                    "macro_delta_rec_vs_M1": macro_vals["delta_rec_vs_M1"],
                    **overall_vals,
                    "mean_cc_count": float(sub["cc_count"].mean()),
                    "mean_small_cc_count": float(sub["small_cc_count"].mean()),
                    "mean_comp_area": float(sub["mean_comp_area"].mean()),
                    "mean_hole_count": float(sub["hole_count"].mean()),
                    "mean_small_hole_count": float(sub["small_hole_count"].mean()),
                    "mean_removed_tp_vs_M0": float(sub["removed_tp_vs_M0"].mean()),
                    "mean_removed_fp_vs_M0": float(sub["removed_fp_vs_M0"].mean()),
                    "mean_added_tp_vs_M0": float(sub["added_tp_vs_M0"].mean()),
                    "mean_added_fp_vs_M0": float(sub["added_fp_vs_M0"].mean()),
                    "mean_removed_tp_vs_M1": float(sub["removed_tp_vs_M1"].mean()),
                    "mean_removed_fp_vs_M1": float(sub["removed_fp_vs_M1"].mean()),
                    "mean_added_tp_vs_M1": float(sub["added_tp_vs_M1"].mean()),
                    "mean_added_fp_vs_M1": float(sub["added_fp_vs_M1"].mean()),
                    "improve_vs_M0_n": int(np.sum(d_m0 >= NEGLIGIBLE_DELTA_THRESHOLD)),
                    "negligible_vs_M0_n": int(np.sum(np.abs(d_m0) < NEGLIGIBLE_DELTA_THRESHOLD)),
                    "regress_vs_M0_n": int(np.sum(d_m0 <= -NEGLIGIBLE_DELTA_THRESHOLD)),
                    "improve_vs_M1_n": int(np.sum(d_m1 >= NEGLIGIBLE_DELTA_THRESHOLD)),
                    "negligible_vs_M1_n": int(np.sum(np.abs(d_m1) < NEGLIGIBLE_DELTA_THRESHOLD)),
                    "regress_vs_M1_n": int(np.sum(d_m1 <= -NEGLIGIBLE_DELTA_THRESHOLD)),
                }
            )
    return pd.DataFrame(rows)


def compute_morph_feature_shifts(
    oof_morph_by_family: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Quantify paired standardized shifts across all 19 Set B morphology features vs M0 and M1."""
    rows: list[dict[str, object]] = []
    for family in ABLATION_FAMILIES:
        arr = oof_morph_by_family[family]  # (N, 7, 19)
        m0_arr = arr[:, ALL_VARIANTS.index("M0"), :]
        m1_arr = arr[:, ALL_VARIANTS.index("M1"), :]
        std_m0 = np.maximum(np.std(m0_arr, axis=0, ddof=0), 1e-12)
        for v_idx, variant in enumerate(ALL_VARIANTS):
            cur = arr[:, v_idx, :]
            delta0 = cur - m0_arr
            delta1 = cur - m1_arr
            z_delta0 = delta0 / std_m0
            for f_idx, fname in enumerate(MORPHOLOGY_FEATURE_NAMES):
                rows.append(
                    {
                        "family": family,
                        "variant": variant,
                        "feature": fname,
                        "mean_value": float(np.mean(cur[:, f_idx])),
                        "std_value": float(np.std(cur[:, f_idx], ddof=0)),
                        "mean_delta_vs_M0": float(np.mean(delta0[:, f_idx])),
                        "std_shift_vs_M0": float(np.mean(z_delta0[:, f_idx])),
                        "mean_abs_std_shift_vs_M0": float(np.mean(np.abs(z_delta0[:, f_idx]))),
                        "mean_abs_std_shift_vs_M1": float(
                            np.mean(np.abs(delta1[:, f_idx]) / std_m0[f_idx])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _assemble_oof_morph_tensor(
    fold_tensor: np.ndarray,
    oof_fold_ids: np.ndarray,
) -> np.ndarray:
    """Gather out-of-fold morphology feature vectors (N, n_var, 19) matching each sample's CV fold."""
    n = fold_tensor.shape[1]
    out = np.empty((n, fold_tensor.shape[2], fold_tensor.shape[3]), dtype=np.float64)
    for idx in range(n):
        out[idx] = fold_tensor[int(oof_fold_ids[idx]), idx]
    return out


def _gamma_value(gamma_str: object) -> float | str:
    txt = str(gamma_str)
    if txt in ("scale", "auto"):
        return txt
    return float(txt)


def evaluate_anchor_cv_grid(
    *,
    train_df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    appearance_train: np.ndarray,
    family_fold_morph: dict[str, np.ndarray],
    val_df: pd.DataFrame | None = None,
    appearance_val: np.ndarray | None = None,
    family_val_morph: dict[str, np.ndarray] | None = None,
    family_full_train_morph: dict[str, np.ndarray] | None = None,
    anchor_specs: tuple[dict[str, object], ...] = PREDECLARED_ANCHOR_SPECS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run strict 5-fold StratifiedGroupKFold SVM evaluation across fixed anchor specs.

    Zero hyperparameter or morphology tuning touches validation data; optional val scores are
    recorded strictly read-only for strict-v3 baseline parity verification.
    """
    y_train = train_df["category"].to_numpy(dtype=object)
    sample_ids = train_df["sample_id"].to_numpy(dtype=object)
    group_ids = train_df["group_id"].to_numpy(dtype=object)
    y_val = val_df["category"].to_numpy(dtype=object) if val_df is not None else None

    b_cols = feature_columns_for_set("B")
    c_cols = feature_columns_for_set("C")
    a_cols = feature_columns_for_set("A")

    cv_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    cat_rows: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []

    eval_targets: list[tuple[str, str, str]] = [("shared_A", "A", "A")]
    for fam in ABLATION_FAMILIES:
        for var in ALL_VARIANTS:
            for fset in ABLATION_FEATURE_SETS:
                eval_targets.append((fam, var, fset))

    for fam, var, fset in eval_targets:
        if fam == "shared_A":
            feat_names = a_cols
            fold_mats = [appearance_train for _ in range(len(folds))]
            full_mat_tr = appearance_train
            X_full_val = appearance_val
            primary_spec_id = PRIMARY_MATCHED_ANCHOR[("shared_A", "A")]
        else:
            v_idx = ALL_VARIANTS.index(var)
            f_tensor = family_fold_morph[fam]  # (folds, N, 7, 19)
            full_morph_tr = (
                family_full_train_morph[fam][:, v_idx, :]
                if family_full_train_morph is not None and fam in family_full_train_morph
                else f_tensor[0, :, v_idx, :]
            )
            if fset == "B":
                feat_names = b_cols
                fold_mats = [f_tensor[f_idx, :, v_idx, :] for f_idx in range(len(folds))]
                full_mat_tr = full_morph_tr
                X_full_val = (
                    family_val_morph[fam][:, v_idx, :]
                    if family_val_morph is not None
                    else None
                )
            else:
                feat_names = c_cols
                fold_mats = [
                    np.hstack([appearance_train, f_tensor[f_idx, :, v_idx, :]])
                    for f_idx in range(len(folds))
                ]
                full_mat_tr = np.hstack([appearance_train, full_morph_tr])
                X_full_val = (
                    np.hstack([appearance_val, family_val_morph[fam][:, v_idx, :]])
                    if (appearance_val is not None and family_val_morph is not None)
                    else None
                )
            primary_spec_id = PRIMARY_MATCHED_ANCHOR[(fam, fset)]

        spec_records: list[dict[str, object]] = []
        for spec in anchor_specs:
            spec_id = str(spec["spec_id"])
            oof_pred = np.empty(len(y_train), dtype=object)
            oof_margin = np.full(len(y_train), np.nan, dtype=np.float64)
            per_fold_metrics: list[dict[str, object]] = []

            for f_idx, (tr_idx, va_idx) in enumerate(folds):
                mat = fold_mats[f_idx]
                scaler = fit_standard_scaler(mat[tr_idx], feat_names)
                X_tr = apply_standard_scaler(mat[tr_idx], scaler)
                X_va = apply_standard_scaler(mat[va_idx], scaler)
                clf = fit_svm_model(
                    X_tr,
                    y_train[tr_idx],
                    kernel=str(spec["kernel"]),
                    C=float(spec["C"]),
                    gamma=_gamma_value(spec["gamma"]),
                    class_weight=str(spec["class_weight"]),
                )
                pred_va = clf.predict(X_va)
                oof_pred[va_idx] = pred_va
                dec = clf.decision_function(X_va)
                if dec.ndim == 2:
                    classes = list(clf.classes_)
                    true_col = np.array([classes.index(lbl) for lbl in y_train[va_idx]], dtype=np.int64)
                    true_scores = dec[np.arange(len(va_idx)), true_col]
                    masked = dec.copy()
                    masked[np.arange(len(va_idx)), true_col] = -np.inf
                    max_other = np.max(masked, axis=1)
                    oof_margin[va_idx] = true_scores - max_other

                fm = classification_metrics(y_train[va_idx], pred_va)
                per_fold_metrics.append(
                    {
                        "fold": f_idx,
                        "macro_f1": float(fm["macro_f1"]),
                        "balanced_accuracy": float(fm["balanced_accuracy"]),
                        "accuracy": float(fm["accuracy"]),
                        "n_fold_val": int(len(va_idx)),
                    }
                )

            pooled = classification_metrics(y_train, oof_pred)
            f_f1s = [r["macro_f1"] for r in per_fold_metrics]

            val_macro_f1 = np.nan
            val_bacc = np.nan
            val_acc = np.nan
            if X_full_val is not None and y_val is not None and len(y_val) > 0:
                full_scaler = fit_standard_scaler(full_mat_tr, feat_names)
                clf_full = fit_svm_model(
                    apply_standard_scaler(full_mat_tr, full_scaler),
                    y_train,
                    kernel=str(spec["kernel"]),
                    C=float(spec["C"]),
                    gamma=_gamma_value(spec["gamma"]),
                    class_weight=str(spec["class_weight"]),
                )
                vm = classification_metrics(
                    y_val,
                    clf_full.predict(apply_standard_scaler(X_full_val, full_scaler)),
                )
                val_macro_f1 = float(vm["macro_f1"])
                val_bacc = float(vm["balanced_accuracy"])
                val_acc = float(vm["accuracy"])

            rec = {
                "family": fam,
                "variant": var,
                "feature_set": fset,
                "spec_id": spec_id,
                "spec_label": str(spec["label"]),
                "is_primary_matched_spec": spec_id == primary_spec_id,
                "kernel": str(spec["kernel"]),
                "class_weight": str(spec["class_weight"]),
                "C": float(spec["C"]),
                "gamma": str(spec["gamma"]),
                "cv_macro_f1": float(pooled["macro_f1"]),
                "cv_fold_macro_f1_mean": float(np.mean(f_f1s)),
                "cv_fold_macro_f1_std": float(np.std(f_f1s, ddof=0)),
                "cv_balanced_accuracy": float(pooled["balanced_accuracy"]),
                "cv_accuracy": float(pooled["accuracy"]),
                "cv_weighted_f1": float(pooled["weighted_f1"]),
                "cv_macro_precision": float(pooled["macro_precision"]),
                "val_macro_f1_readonly": val_macro_f1,
                "val_balanced_accuracy_readonly": val_bacc,
                "val_accuracy_readonly": val_acc,
                "_oof_pred": oof_pred,
                "_oof_margin": oof_margin,
                "_per_fold": per_fold_metrics,
            }
            spec_records.append(rec)

        best_rec = max(spec_records, key=cv_selection_key_for_svm)
        for rec in spec_records:
            is_best = rec is best_rec
            oof_pred = rec.pop("_oof_pred")
            oof_margin = rec.pop("_oof_margin")
            per_fold_metrics = rec.pop("_per_fold")
            rec["is_best_anchor_cv"] = is_best
            cv_rows.append(rec)

            for pf in per_fold_metrics:
                fold_rows.append(
                    {
                        "family": fam,
                        "variant": var,
                        "feature_set": fset,
                        "spec_id": rec["spec_id"],
                        "is_primary_matched_spec": rec["is_primary_matched_spec"],
                        "is_best_anchor_cv": is_best,
                        **pf,
                    }
                )

            # Save OOF sample predictions & per-category tables for key matched / best specs.
            keep_detail = (
                bool(rec["is_primary_matched_spec"])
                or bool(is_best)
                or rec["spec_id"] in ("lin_none_C3", "rbf_bal_C3_g001")
            )
            if keep_detail:
                cat_tbl = per_category_classification_table(y_train, oof_pred)
                for c_rec in cat_tbl.to_dict("records"):
                    cat_rows.append(
                        {
                            "family": fam,
                            "variant": var,
                            "feature_set": fset,
                            "spec_id": rec["spec_id"],
                            "is_primary_matched_spec": rec["is_primary_matched_spec"],
                            "is_best_anchor_cv": is_best,
                            **c_rec,
                        }
                    )
                if bool(rec["is_primary_matched_spec"]) and fam != "shared_A":
                    for i in range(len(y_train)):
                        oof_rows.append(
                            {
                                "family": fam,
                                "variant": var,
                                "feature_set": fset,
                                "spec_id": rec["spec_id"],
                                "sample_id": str(sample_ids[i]),
                                "group_id": str(group_ids[i]),
                                "y_true": str(y_train[i]),
                                "y_pred": str(oof_pred[i]),
                                "is_correct": bool(oof_pred[i] == y_train[i]),
                                "true_margin": float(oof_margin[i]),
                            }
                        )

    cv_df = pd.DataFrame(cv_rows)
    fold_df = _attach_paired_fold_deltas(pd.DataFrame(fold_rows))
    cat_df = pd.DataFrame(cat_rows)
    oof_df = pd.DataFrame(oof_rows)
    return cv_df, fold_df, cat_df, oof_df


def _attach_paired_fold_deltas(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Attach exact per-fold paired Macro-F1 differences vs M0, M1, and M2 baselines."""
    base_lookup = (
        fold_df.loc[fold_df["variant"].isin(BASELINE_VARIANTS)]
        .set_index(["family", "feature_set", "spec_id", "fold", "variant"])["macro_f1"]
        .to_dict()
    )
    out = fold_df.copy()
    for b_var in BASELINE_VARIANTS:
        deltas = []
        for r in out.itertuples(index=False):
            if r.family == "shared_A":
                deltas.append(np.nan)
            else:
                b_val = base_lookup[(r.family, r.feature_set, r.spec_id, r.fold, b_var)]
                deltas.append(float(r.macro_f1) - float(b_val))
        out[f"delta_f1_vs_{b_var}"] = deltas
    return out


def select_deterministic_ablation_cases(
    seg_train_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    family: str = "otsu",
) -> pd.DataFrame:
    """Deterministically select representative segmentation and OOF Dice-up/Cls-down cases."""
    sub_seg = seg_train_df.loc[seg_train_df["family"].eq(family)].copy()
    piv_dice = sub_seg.pivot(index="sample_id", columns="variant", values="dice")
    meta = (
        sub_seg.loc[sub_seg["variant"].eq("M0"), ["sample_id", "category", "group_id"]]
        .set_index("sample_id")
    )
    meta["dice_M0"] = piv_dice["M0"]
    meta["dice_M1"] = piv_dice["M1"]
    meta["dice_M2"] = piv_dice["M2"]
    meta["dice_close"] = piv_dice["close_only"]
    meta["dice_m2a16"] = piv_dice["m2_area16"]
    meta["dice_hole"] = piv_dice["hole_fill_small"]
    meta["delta_close_M0"] = meta["dice_close"] - meta["dice_M0"]
    meta["delta_open_M1"] = piv_dice["open_only"] - meta["dice_M1"]
    meta["delta_m2a16_M2"] = meta["dice_m2a16"] - meta["dice_M2"]
    meta["delta_hole_M1"] = meta["dice_hole"] - meta["dice_M1"]

    oof_sub = oof_df.loc[oof_df["family"].eq(family) & oof_df["feature_set"].eq("C")].copy()
    piv_ok = oof_sub.pivot(index="sample_id", columns="variant", values="is_correct")
    piv_mg = oof_sub.pivot(index="sample_id", columns="variant", values="true_margin")
    for var in ALL_VARIANTS:
        if var in piv_ok.columns:
            meta[f"ok_{var}"] = piv_ok[var]
            meta[f"margin_{var}"] = piv_mg[var]

    meta = meta.reset_index()
    picks: list[dict[str, object]] = []

    # 1. Largest CLOSE-only gain vs M0 where OPEN-only drops on Polyp or overall
    p1 = meta.sort_values(["delta_close_M0", "sample_id"], ascending=[False, True]).iloc[0]
    picks.append({"role": "Max Dice Gain (CLOSE-only vs M0)", "target_variant": "close_only", **p1})

    # 2. Largest OPEN-only erosion regression on Polyp vs M1
    polyp_only = meta.loc[meta["category"].eq("Polyp")]
    if polyp_only.empty:
        polyp_only = meta
    p2 = polyp_only.sort_values(["delta_open_M1", "sample_id"], ascending=[True, True]).iloc[0]
    picks.append({"role": "Polyp Erosion (OPEN-only vs M1)", "target_variant": "open_only", **p2})

    # 3. Largest HoleFill gain vs M1
    p3 = meta.sort_values(["delta_hole_M1", "sample_id"], ascending=[False, True]).iloc[0]
    picks.append({"role": "Max HoleFill Gain (vs M1)", "target_variant": "hole_fill_small", **p3})

    # 4. Deterministic case where morphology raises Dice (>= +0.005) vs M0 but OOF Set C flips correct->wrong
    for cand_var, col in (("M2", "dice_M2"), ("close_only", "dice_close"), ("m2_area16", "dice_m2a16")):
        if f"ok_{cand_var}" in meta.columns and "ok_M0" in meta.columns:
            flip = meta.loc[
                (meta[col] - meta["dice_M0"] >= 0.01)
                & meta["ok_M0"].eq(True)
                & meta[f"ok_{cand_var}"].eq(False)
            ].copy()
            if not flip.empty:
                flip["_d_dice"] = flip[col] - flip["dice_M0"]
                flip["_d_margin"] = flip[f"margin_{cand_var}"] - flip["margin_M0"]
                picked = flip.sort_values(["_d_margin", "_d_dice", "sample_id"], ascending=[True, False, True]).iloc[0]
                picks.append(
                    {
                        "role": f"Dice Up / Cls Flip ({cand_var} vs M0)",
                        "target_variant": cand_var,
                        **{k: v for k, v in picked.items() if not k.startswith("_")},
                    }
                )
                break

    out = pd.DataFrame(picks)
    cols = [
        "role",
        "target_variant",
        "sample_id",
        "category",
        "group_id",
        "dice_M0",
        "dice_M1",
        "dice_M2",
        "dice_close",
        "dice_m2a16",
        "dice_hole",
        "ok_M0",
        "ok_M1",
        "ok_M2",
        "ok_close_only",
        "ok_m2_area16",
        "ok_hole_fill_small",
    ]
    keep_cols = [c for c in cols if c in out.columns]
    return out[keep_cols].drop_duplicates(subset=["role", "sample_id"]).reset_index(drop=True)


def run_controlled_morphology_ablation(
    project_root: str | Path,
    *,
    max_workers: int | None = None,
    use_cache: bool = True,
    save_outputs: bool = True,
) -> AblationExperimentOutputs:
    """Execute the full Task 2 controlled morphology ablation on train Strict-CV + read-only val."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    manifest_df = load_split_manifest(root, verify=True)

    train_app_df, val_app_df, app_sha = load_shared_appearance_frame(root)
    if app_sha != split_sha:
        raise ValueError("Frozen split SHA mismatch with shared appearance features.")

    train_df, folds, oof_fold_ids = build_strict_train_folds(manifest_df, n_splits=CV_FOLDS)
    # Align shared appearance matrices explicitly by sample_id order.
    app_train_mat = (
        train_app_df.set_index("sample_id")
        .loc[train_df["sample_id"].tolist(), list(APPEARANCE_FEATURE_NAMES)]
        .to_numpy(dtype=np.float64)
    )
    val_df_sorted = (
        manifest_df.loc[manifest_df["split"].eq("val")]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    app_val_mat = (
        val_app_df.set_index("sample_id")
        .loc[val_df_sorted["sample_id"].tolist(), list(APPEARANCE_FEATURE_NAMES)]
        .to_numpy(dtype=np.float64)
    )

    workers = max_workers if max_workers is not None else min(12, max(1, (os.cpu_count() or 2) - 2))
    otsu_fits, k4_fits = _resolve_fold_upstream_models(root, train_df, folds)

    otsu_fold_morph, k4_fold_morph, seg_train_df, area_dist_df = _load_or_compute_train_features_and_seg(
        root,
        train_df,
        otsu_fits,
        k4_fits,
        oof_fold_ids,
        max_workers=workers,
        use_ablation_cache=use_cache,
    )

    val_otsu_morph, val_k4_morph, full_train_k4_morph, seg_val_df = (
        _compute_full_train_k4_and_val_readonly(
            root, train_df, manifest_df, max_workers=workers
        )
    )
    print(f"  [ablation] val read-only & full-train k4 done ({len(val_df_sorted)} val imgs)", flush=True)

    seg_train_summary = summarize_ablation_segmentation(seg_train_df)
    seg_val_summary = summarize_ablation_segmentation(seg_val_df)

    oof_morph_map = {
        "otsu": _assemble_oof_morph_tensor(otsu_fold_morph, oof_fold_ids),
        "k4": _assemble_oof_morph_tensor(k4_fold_morph, oof_fold_ids),
    }
    feature_shifts = compute_morph_feature_shifts(oof_morph_map)

    cv_df, fold_df, cat_df, oof_df = evaluate_anchor_cv_grid(
        train_df=train_df,
        folds=folds,
        appearance_train=app_train_mat,
        family_fold_morph={"otsu": otsu_fold_morph, "k4": k4_fold_morph},
        val_df=val_df_sorted,
        appearance_val=app_val_mat,
        family_val_morph={"otsu": val_otsu_morph, "k4": val_k4_morph},
        family_full_train_morph={
            "otsu": otsu_fold_morph[0],
            "k4": full_train_k4_morph,
        },
    )
    print(f"  [ablation] strict-CV SVMs done ({len(cv_df)} config-spec rows)", flush=True)

    cases_df = select_deterministic_ablation_cases(seg_train_df, oof_df, family="otsu")

    manifest: dict[str, object] = {
        "schema_version": "ebhi_dip.task2_controlled_morphology_ablation.v1",
        "split_sha256": split_sha,
        "train_sample_ids_sha256": _train_sample_ids_sha(train_df["sample_id"].tolist()),
        "train_images": int(len(train_df)),
        "train_groups": int(train_df["group_id"].nunique()),
        "val_readonly_images": int(len(val_df_sorted)),
        "val_readonly_groups": int(val_df_sorted["group_id"].nunique()),
        "test_images_touched": 0,
        "test_labels_accessed": False,
        "cv_folds": CV_FOLDS,
        "cv_scheme": "StratifiedGroupKFold",
        "min_component_area": MIN_COMPONENT_AREA,
        "max_small_hole_area_predeclared": MAX_SMALL_HOLE_AREA,
        "hole_connectivity": 4,
        "component_connectivity": 8,
        "cross_kernel": CROSS_3X3_KERNEL.tolist(),
        "otsu_fold_polarities": [str(f["chosen_polarity"]) for f in otsu_fits],
        "variants": list(ALL_VARIANTS),
        "families": list(ABLATION_FAMILIES),
        "anchor_specs": list(PREDECLARED_ANCHOR_SPECS),
        "kmeans_seed": 42,
        "versions": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
            "scikit_learn": sklearn.__version__,
            "scikit_image": skimage.__version__,
        },
        "protected_input_sha256": {
            "split_csv": sha256_bytes(root / FROZEN_SPLIT_REL),
            "split_sha": sha256_bytes(root / FROZEN_SPLIT_SHA_REL),
            "otsu_raw_json": sha256_bytes(root / "configs/otsu_raw.json"),
            "kmeans_k4_json": sha256_bytes(root / "models/kmeans_lab_k4_mapped.json"),
            "appearance_csv": sha256_bytes(root / "results/features/appearance.csv"),
        },
    }

    outputs = AblationExperimentOutputs(
        manifest=manifest,
        seg_train_per_image=seg_train_df,
        seg_train_summary=seg_train_summary,
        seg_val_summary=seg_val_summary,
        area_distributions=area_dist_df,
        feature_shifts=feature_shifts,
        cv_results=cv_df,
        fold_deltas=fold_df,
        per_category_cv=cat_df,
        oof_predictions=oof_df,
        deterministic_cases=cases_df,
    )

    if save_outputs:
        save_ablation_outputs(
            root,
            outputs,
            otsu_fold_morph=otsu_fold_morph,
            k4_fold_morph=k4_fold_morph,
            train_ids_sha=str(manifest["train_sample_ids_sha256"]),
        )
    return outputs


def save_ablation_outputs(
    project_root: str | Path,
    outputs: AblationExperimentOutputs,
    *,
    otsu_fold_morph: np.ndarray | None = None,
    k4_fold_morph: np.ndarray | None = None,
    train_ids_sha: str | None = None,
) -> Path:
    """Save Task 2 outputs exclusively under results/morphology_ablation/ without touching v1-v3."""
    out_dir = Path(project_root).expanduser().resolve() / RESULTS_ABLATION_DIR_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs.seg_train_per_image.to_csv(out_dir / "segmentation_train_oof_per_image.csv", index=False)
    outputs.seg_train_summary.to_csv(out_dir / "segmentation_train_oof_summary.csv", index=False)
    outputs.seg_val_summary.to_csv(out_dir / "segmentation_val_readonly_summary.csv", index=False)
    outputs.area_distributions.to_csv(out_dir / "component_area_distributions_train_oof.csv", index=False)
    outputs.feature_shifts.to_csv(out_dir / "feature_shift_train_oof.csv", index=False)
    outputs.cv_results.to_csv(out_dir / "svm_ablation_cv_results.csv", index=False)
    outputs.fold_deltas.to_csv(out_dir / "svm_ablation_fold_deltas.csv", index=False)
    outputs.per_category_cv.to_csv(out_dir / "svm_ablation_per_category_cv.csv", index=False)
    outputs.oof_predictions.to_csv(out_dir / "svm_ablation_oof_predictions.csv", index=False)
    outputs.deterministic_cases.to_csv(out_dir / "deterministic_case_records.csv", index=False)

    if otsu_fold_morph is not None and k4_fold_morph is not None and train_ids_sha is not None:
        np.savez_compressed(
            out_dir / CACHE_ABLATION_FILENAME,
            train_sample_ids_sha256=np.array(train_ids_sha),
            variants=np.array(ALL_VARIANTS),
            max_small_hole_area=np.array(MAX_SMALL_HOLE_AREA, dtype=np.int64),
            otsu_morph=otsu_fold_morph,
            k4_morph=k4_fold_morph,
        )

    (out_dir / "ablation_manifest.json").write_text(
        json.dumps(outputs.manifest, indent=2),
        encoding="utf-8",
    )
    return out_dir


def load_saved_ablation_outputs(project_root: str | Path) -> AblationExperimentOutputs:
    """Load saved Task 2 results cleanly for fast visual inspection and reporting."""
    out_dir = Path(project_root).expanduser().resolve() / RESULTS_ABLATION_DIR_REL
    manifest = json.loads((out_dir / "ablation_manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("test_images_touched", -1)) != 0 or manifest.get("test_labels_accessed") is not False:
        raise ValueError("Invalid provenance in saved ablation manifest.")
    return AblationExperimentOutputs(
        manifest=manifest,
        seg_train_per_image=pd.read_csv(out_dir / "segmentation_train_oof_per_image.csv"),
        seg_train_summary=pd.read_csv(out_dir / "segmentation_train_oof_summary.csv"),
        seg_val_summary=pd.read_csv(out_dir / "segmentation_val_readonly_summary.csv"),
        area_distributions=pd.read_csv(out_dir / "component_area_distributions_train_oof.csv"),
        feature_shifts=pd.read_csv(out_dir / "feature_shift_train_oof.csv"),
        cv_results=pd.read_csv(out_dir / "svm_ablation_cv_results.csv"),
        fold_deltas=pd.read_csv(out_dir / "svm_ablation_fold_deltas.csv"),
        per_category_cv=pd.read_csv(out_dir / "svm_ablation_per_category_cv.csv"),
        oof_predictions=pd.read_csv(out_dir / "svm_ablation_oof_predictions.csv"),
        deterministic_cases=pd.read_csv(out_dir / "deterministic_case_records.csv"),
    )


def plot_ablation_segmentation_overview(seg_summary_df: pd.DataFrame) -> Figure:
    """Compare Macro-Dice, Foreground Coverage (%), and Precision/Recall shifts across 7 variants."""
    configure_notebook_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), layout="constrained")
    x = np.arange(len(ALL_VARIANTS), dtype=np.float64)
    w = 0.36
    x_lbls = [VARIANT_SHORT_LABELS[v] for v in ALL_VARIANTS]

    o_sub = seg_summary_df.loc[seg_summary_df["family"].eq("otsu")].set_index("variant").loc[list(ALL_VARIANTS)]
    k_sub = seg_summary_df.loc[seg_summary_df["family"].eq("k4")].set_index("variant").loc[list(ALL_VARIANTS)]

    # Panel 0: Macro-Dice
    ax0 = axes[0]
    b0 = ax0.bar(x - w / 2, o_sub["macro_dice"], width=w, color="#1F77B4", label="Otsu")
    b1 = ax0.bar(x + w / 2, k_sub["macro_dice"], width=w, color="#D95F02", label="K-Means k=4")
    ax0.axhline(float(o_sub.loc["M0", "macro_dice"]), color="#1F77B4", linestyle=":", linewidth=1.1)
    ax0.axhline(float(k_sub.loc["M0", "macro_dice"]), color="#D95F02", linestyle=":", linewidth=1.1)
    for bar in (*b0, *b1):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7.0, rotation=45)
    ax0.set_ylim(0.55, 0.88)
    ax0.set_xticks(x)
    ax0.set_xticklabels(x_lbls, rotation=22, ha="right", fontsize=8.3)
    ax0.set_ylabel("Makro-Dice (6 kategori, train OOF)")
    ax0.set_title("Kualitas Segmentasi (Makro-Dice)", fontweight="bold", fontsize=9.8)
    ax0.grid(axis="y", alpha=0.28)
    ax0.legend(loc="lower right", fontsize=8.0, frameon=True)

    # Panel 1: Mean Foreground Coverage %
    ax1 = axes[1]
    ax1.bar(x - w / 2, o_sub["macro_fg_pct"], width=w, color="#5DADE2", label="Otsu")
    ax1.bar(x + w / 2, k_sub["macro_fg_pct"], width=w, color="#F5B041", label="K-Means k=4")
    ax1.set_ylim(0, 102)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_lbls, rotation=22, ha="right", fontsize=8.3)
    ax1.set_ylabel("Cakupan Foreground (%)")
    ax1.set_title("Rata-rata Cakupan Foreground", fontweight="bold", fontsize=9.8)
    ax1.grid(axis="y", alpha=0.28)
    ax1.legend(fontsize=8.0, frameon=True)

    # Panel 2: Precision vs Recall shifts relative to M0 (Otsu & k4)
    ax2 = axes[2]
    ax2.axhline(0.0, color="#7F8C8D", linestyle="--", linewidth=0.9)
    ax2.axvline(0.0, color="#7F8C8D", linestyle="--", linewidth=0.9)
    label_offsets = {
        ("k4", "M1"): (5, 3, "left"),
        ("k4", "M2"): (6, 4, "left"),
        ("k4", "close_only"): (5, 3, "left"),
        ("k4", "open_only"): (5, 2, "left"),
        ("k4", "m2_area16"): (-6, -9, "right"),
        ("k4", "hole_fill_small"): (5, 4, "left"),
        ("otsu", "M1"): (5, 2, "left"),
        ("otsu", "M2"): (5, 3, "left"),
        ("otsu", "close_only"): (5, 2, "left"),
        ("otsu", "open_only"): (-6, 4, "right"),
        ("otsu", "m2_area16"): (5, -8, "left"),
        ("otsu", "hole_fill_small"): (5, 3, "left"),
    }
    for fam, marker, lbl_f in (("otsu", "o", "Otsu"), ("k4", "^", "k4")):
        f_sub = seg_summary_df.loc[seg_summary_df["family"].eq(fam)].set_index("variant")
        base_p = float(f_sub.loc["M0", "macro_precision"])
        base_r = float(f_sub.loc["M0", "macro_recall"])
        for var in ALL_VARIANTS[1:]:
            dp = float(f_sub.loc[var, "macro_precision"]) - base_p
            dr = float(f_sub.loc[var, "macro_recall"]) - base_r
            ax2.scatter(dp, dr, s=68, marker=marker, color=VARIANT_COLORS[var], edgecolor="#1C2833", linewidth=0.7, zorder=4)
            ox, oy, ha_mode = label_offsets.get((fam, var), (4, 3, "left"))
            ax2.annotate(
                f"{lbl_f}:{VARIANT_SHORT_LABELS[var]}",
                (dp, dr),
                xytext=(ox, oy),
                textcoords="offset points",
                ha=ha_mode,
                fontsize=7.0,
                color="#222222",
            )
    ax2.margins(x=0.18, y=0.14)
    ax2.set_xlabel("ΔPrecision vs M0")
    ax2.set_ylabel("ΔRecall vs M0")
    ax2.set_title("Trade-off Precision vs Recall", fontweight="bold", fontsize=9.8)
    ax2.grid(alpha=0.28)
    return fig


def plot_ablation_delta_dice_distributions(seg_train_df: pd.DataFrame) -> Figure:
    """Plot per-image paired ΔDice vs M1 across all post-M1 operations (Otsu vs k4)."""
    configure_notebook_style()
    compare_vars = ("M2", "close_only", "open_only", "m2_area16", "hole_fill_small")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.4), layout="constrained", sharey=True)

    for ax_idx, (fam, title_fam) in enumerate((("otsu", "Otsu"), ("k4", "K-Means k=4"))):
        ax = axes[ax_idx]
        sub = seg_train_df.loc[seg_train_df["family"].eq(fam)]
        data = [sub.loc[sub["variant"].eq(v), "delta_dice_vs_M1"].to_numpy() for v in compare_vars]
        ax.axhspan(-NEGLIGIBLE_DELTA_THRESHOLD, NEGLIGIBLE_DELTA_THRESHOLD, color="#EAEDED", alpha=0.85, label="|Δ| < 0.005")
        ax.axhline(0.0, color="#566573", linestyle="--", linewidth=1.0)
        bp = ax.boxplot(data, patch_artist=True, showfliers=True, flierprops={"markersize": 2.6, "alpha": 0.35})
        for idx, box in enumerate(bp["boxes"]):
            box.set_facecolor(VARIANT_COLORS[compare_vars[idx]])
            box.set_alpha(0.78)
        for med in bp["medians"]:
            med.set_color("#111111")
            med.set_linewidth(1.5)
        ax.set_xticks(np.arange(1, len(compare_vars) + 1))
        ax.set_xticklabels([VARIANT_SHORT_LABELS[v] for v in compare_vars], fontsize=8.5)
        ax.set_ylim(-0.08, 0.08)
        ax.set_title(f"Distribusi ΔDice vs M1 — {title_fam} (train OOF, n=1.535)", fontweight="bold", fontsize=9.8)
        if ax_idx == 0:
            ax.set_ylabel("ΔDice Berpasangan (Varian − M1)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=8.0, frameon=True)
    return fig


def plot_ablation_pixel_flips_and_components(seg_summary_df: pd.DataFrame) -> Figure:
    """Plot TP/FP pixels added or removed relative to M1 alongside component counts & <16 px remnants."""
    configure_notebook_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.6), layout="constrained")
    post_vars = ("M2", "close_only", "open_only", "m2_area16", "hole_fill_small")

    # Left panel: Otsu mean TP/FP pixels added (+) and removed (-) vs M1 per image
    ax0 = axes[0]
    o_sub = seg_summary_df.loc[seg_summary_df["family"].eq("otsu")].set_index("variant").loc[list(post_vars)]
    px = np.arange(len(post_vars), dtype=np.float64)
    w = 0.36
    add_tp = o_sub["mean_added_tp_vs_M1"].to_numpy()
    add_fp = o_sub["mean_added_fp_vs_M1"].to_numpy()
    rem_tp = -o_sub["mean_removed_tp_vs_M1"].to_numpy()
    rem_fp = -o_sub["mean_removed_fp_vs_M1"].to_numpy()

    ax0.bar(px, add_tp, width=0.52, color="#2EA043", alpha=0.88, label="+TP dibanding M1")
    ax0.bar(px, add_fp, width=0.52, bottom=add_tp, color="#E67E22", alpha=0.88, label="+FP dibanding M1")
    ax0.bar(px, rem_tp, width=0.52, color="#8E44AD", alpha=0.88, label="−TP hilang dibanding M1")
    ax0.bar(px, rem_fp, width=0.52, bottom=rem_tp, color="#3498DB", alpha=0.88, label="−FP terhapus dibanding M1")
    ax0.axhline(0.0, color="#2C3E50", linewidth=1.0)
    ax0.set_xticks(px)
    ax0.set_xticklabels([VARIANT_SHORT_LABELS[v] for v in post_vars], fontsize=8.6)
    ax0.set_ylabel("Rata-rata Piksel Berubah / Citra")
    ax0.set_title("Perubahan Piksel TP vs FP dibanding M1 (Otsu)", fontweight="bold", fontsize=9.8)
    ax0.set_ylim(float(np.min(rem_tp + rem_fp)) * 1.15, float(np.max(add_tp + add_fp)) * 1.24)
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="upper right", ncol=2, fontsize=7.8, frameon=True)

    # Right panel: Total connected components vs <16 px small components (Otsu train OOF)
    ax1 = axes[1]
    o_all = seg_summary_df.loc[seg_summary_df["family"].eq("otsu")].set_index("variant").loc[list(ALL_VARIANTS)]
    cx = np.arange(len(ALL_VARIANTS), dtype=np.float64)
    large_cc = (o_all["mean_cc_count"] - o_all["mean_small_cc_count"]).to_numpy()
    small_cc = o_all["mean_small_cc_count"].to_numpy()
    ax1.bar(cx, large_cc, width=0.55, color="#34495E", label="Komponen ≥ 16 px")
    ax1.bar(cx, small_cc, width=0.55, bottom=large_cc, color="#DC3545", label="Fragmen < 16 px")
    for idx, (lg, sm) in enumerate(zip(large_cc, small_cc)):
        ax1.text(idx, lg + sm + 2.5, f"{lg + sm:.1f}\n({sm:.1f}<16)", ha="center", va="bottom", fontsize=7.2)
    ax1.set_ylim(0, float(np.max(large_cc + small_cc)) * 1.18)
    ax1.set_xticks(cx)
    ax1.set_xticklabels([VARIANT_SHORT_LABELS[v] for v in ALL_VARIANTS], rotation=20, ha="right", fontsize=8.4)
    ax1.set_ylabel("Rata-rata Komponen / Citra")
    ax1.set_title("Jumlah Komponen & Sisa Fragmen < 16 px (Otsu)", fontweight="bold", fontsize=9.8)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8.0, frameon=True)
    return fig


def plot_ablation_feature_shifts(feature_shift_df: pd.DataFrame, family: str = "otsu") -> Figure:
    """Plot annotated heatmap of standardized mean |Δx|/σ_M0 across the 19 Set B morphology features."""
    configure_notebook_style()
    vars_post = list(ALL_VARIANTS[1:])
    sub = feature_shift_df.loc[
        feature_shift_df["family"].eq(family) & feature_shift_df["variant"].isin(vars_post)
    ]
    mat = (
        sub.pivot(index="feature", columns="variant", values="mean_abs_std_shift_vs_M0")
        .loc[list(MORPHOLOGY_FEATURE_NAMES), vars_post]
    )
    vals = mat.to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10.2, 7.0), layout="constrained")
    im = ax.imshow(vals, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(1.2, float(np.max(vals))))
    ax.set_xticks(np.arange(len(vars_post)))
    ax.set_xticklabels([VARIANT_SHORT_LABELS[v] for v in vars_post], fontsize=9.0, fontweight="bold")
    ax.set_yticks(np.arange(len(MORPHOLOGY_FEATURE_NAMES)))
    ax.set_yticklabels(list(MORPHOLOGY_FEATURE_NAMES), fontsize=8.2)
    for r in range(vals.shape[0]):
        for c in range(vals.shape[1]):
            v = vals[r, c]
            r_c, g_c, b_c, _ = im.cmap(im.norm(v))
            luminance = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
            txt_col = "white" if luminance < 0.50 else "#1C2833"
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color=txt_col)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Rata-rata |Δfitur| / σ_M0", fontsize=9.0)
    fam_title = "Otsu" if family == "otsu" else "K-Means k=4"
    ax.set_title(f"Pergeseran Terstandar 19 Fitur Set B dibanding M0 ({fam_title}, train OOF)", fontweight="bold", fontsize=10.5)
    return fig


def plot_ablation_cv_macro_f1(
    cv_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> Figure:
    """Compare 5-fold StratifiedGroupKFold Macro-F1 across M0/M1/M2 and the 4 new candidates."""
    configure_notebook_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8), layout="constrained", sharey=True)

    shared_best = float(
        cv_df.loc[cv_df["family"].eq("shared_A") & cv_df["is_best_anchor_cv"], "cv_macro_f1"].iloc[0]
    )
    otsu_m0_c_lin = float(
        cv_df.loc[
            cv_df["family"].eq("otsu")
            & cv_df["variant"].eq("M0")
            & cv_df["feature_set"].eq("C")
            & cv_df["spec_id"].eq("lin_none_C3"),
            "cv_macro_f1",
        ].iloc[0]
    )

    panels = (
        ("otsu", "rbf_bal_C10_g001", "rbf_bal_C3_g001", "lin_none_C3", "Otsu (Anchor Ketat v3)"),
        ("k4", "rbf_bal_C3_scale", "rbf_bal_C3_scale", "lin_none_C3", "K-Means k=4 (Anchor Ketat v3)"),
    )
    available_specs = set(cv_df["spec_id"].astype(str).unique())
    fallback_spec = sorted(available_specs)[0]
    x = np.arange(len(ALL_VARIANTS), dtype=np.float64)
    w = 0.25

    for ax_idx, (fam, spec_b, spec_c1, spec_c2, title_txt) in enumerate(panels):
        ax = axes[ax_idx]
        ax.axhline(shared_best, color="#7F8C8D", linestyle="--", linewidth=1.2, label=f"shared_A ({shared_best:.4f})")
        if fam == "otsu":
            ax.axhline(otsu_m0_c_lin, color="#111111", linestyle=":", linewidth=1.2, label=f"Otsu M0-C Lin ({otsu_m0_c_lin:.4f})")

        sp_b = spec_b if spec_b in available_specs else fallback_spec
        sp_c1 = spec_c1 if spec_c1 in available_specs else fallback_spec
        sp_c2 = spec_c2 if spec_c2 in available_specs else fallback_spec
        series_specs = [
            ("B", sp_b, f"Set B ({sp_b})", "#5DADE2"),
            ("C", sp_c1, f"Set C ({sp_c1})", "#1B9E77"),
            ("C", sp_c2, f"Set C ({sp_c2})", "#D95F02"),
        ]
        if sp_c2 == sp_c1:
            series_specs = series_specs[:2]
        for s_idx, (fset, sp_id, lbl, color) in enumerate(series_specs):
            sub = (
                cv_df.loc[
                    cv_df["family"].eq(fam)
                    & cv_df["feature_set"].eq(fset)
                    & cv_df["spec_id"].eq(sp_id)
                ]
                .set_index("variant")
                .loc[list(ALL_VARIANTS)]
            )
            offsets = x + (s_idx - (len(series_specs) - 1) / 2.0) * w
            vals = sub["cv_macro_f1"].to_numpy()
            bars = ax.bar(offsets, vals, width=w, color=color, alpha=0.88, label=lbl)

            # Plot per-fold min..max error whiskers around bar tops for fold stability visibility.
            f_sub = fold_df.loc[
                fold_df["family"].eq(fam)
                & fold_df["feature_set"].eq(fset)
                & fold_df["spec_id"].eq(sp_id)
            ]
            f_min = f_sub.groupby("variant")["macro_f1"].min().loc[list(ALL_VARIANTS)].to_numpy()
            f_max = f_sub.groupby("variant")["macro_f1"].max().loc[list(ALL_VARIANTS)].to_numpy()
            for x_pos, y_v, lo, hi in zip(offsets, vals, f_min, f_max):
                ax.plot([x_pos, x_pos], [lo, hi], color="#1C2833", linewidth=1.0, alpha=0.7)
            for bar, hi in zip(bars, f_max):
                top_y = max(float(bar.get_height()), float(hi)) + 0.006
                ax.text(bar.get_x() + bar.get_width() / 2, top_y, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_SHORT_LABELS[v] for v in ALL_VARIANTS], rotation=20, ha="right", fontsize=8.4)
        ax.set_ylim(0.20, 0.68)
        ax.set_title(title_txt, fontweight="bold", fontsize=9.8)
        if ax_idx == 0:
            ax.set_ylabel("Grouped-CV Macro-F1 (Pooled OOF)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", ncol=2 if fam == "otsu" else 1, fontsize=7.5, frameon=True)
    return fig


def plot_ablation_per_category_f1(
    per_cat_df: pd.DataFrame,
    *,
    family: str = "otsu",
    feature_set: str = "C",
    spec_id: str = "rbf_bal_C3_g001",
) -> Figure:
    """Compare pooled OOF per-category F1 across all 7 morphology configurations."""
    configure_notebook_style()
    sub = per_cat_df.loc[
        per_cat_df["family"].eq(family)
        & per_cat_df["feature_set"].eq(feature_set)
        & per_cat_df["spec_id"].eq(spec_id)
    ]
    mat = sub.pivot(index="category", columns="variant", values="f1").loc[
        list(EXPECTED_CATEGORIES), list(ALL_VARIANTS)
    ]
    cats = list(EXPECTED_CATEGORIES)
    cx = np.arange(len(cats), dtype=np.float64)
    w = 0.11

    fig, ax = plt.subplots(figsize=(13.8, 4.6), layout="constrained")
    for idx, var in enumerate(ALL_VARIANTS):
        vals = mat[var].to_numpy(dtype=np.float64)
        ax.bar(cx + (idx - 3) * w, vals, width=w, color=VARIANT_COLORS[var], label=VARIANT_SHORT_LABELS[var])

    ax.set_xticks(cx)
    ax.set_xticklabels(cats, rotation=15, ha="right", fontsize=8.8)
    ax.set_ylabel("Out-of-Fold Category F1")
    ax.set_ylim(0.0, 0.85)
    ax.set_title(
        f"Konsistensi F1 per Kategori — {family.upper()} Set {feature_set} ({spec_id}, 5-fold CV)",
        fontweight="bold",
        fontsize=10.2,
    )
    ax.grid(axis="y", alpha=0.28)
    ax.legend(ncol=7, loc="upper right", fontsize=8.0, frameon=True)
    return fig


def plot_case_ablation_gallery(
    project_root: str | Path,
    sample_id: str,
    *,
    family: str = "otsu",
    title_prefix: str = "",
) -> Figure:
    """Plot RGB+GT alongside M0, M1, M2 and the 4 new morphology masks with error overlays."""
    configure_notebook_style()
    root = Path(project_root).expanduser().resolve()
    manifest = load_split_manifest(root, verify=True)
    row = manifest.loc[manifest["sample_id"].eq(sample_id)].iloc[0]
    if row["split"] == "test":
        raise ValueError(f"Forbidden test sample: {sample_id}")

    rgb = load_rgb(root / str(row["image_path"]))
    ref = load_reference_mask(root / str(row["mask_path"]))
    if family == "otsu":
        otsu_cfg = load_otsu_config(root)
        _, raw_m0 = otsu_threshold_mask(rgb, foreground_polarity=str(otsu_cfg["foreground_polarity"]))
    else:
        km = load_kmeans_model(root, k=4, verify_provenance=True)
        lbl = nearest_clusters(rgb_to_lab(rgb).reshape(-1, 3), km["cluster_centers"]).reshape(rgb.shape[:2])
        raw_m0 = np.isin(lbl, km["foreground_clusters"])

    masks = ablation_morphology_variants(raw_m0)
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.2), layout="constrained")
    panels: list[tuple[str, np.ndarray, str]] = [
        ("RGB + Kontur GT", render_rgb_with_gt_contour(rgb, ref), f"{sample_id.split('/', 1)[1]} ({row['category']})"),
    ]
    for var in ALL_VARIANTS:
        m = masks[var]
        met = binary_segmentation_metrics(m, ref)
        ov, _ = build_error_overlay(rgb, m, ref, alpha=0.48)
        cc = len(connected_component_areas(m))
        panels.append(
            (VARIANT_SHORT_LABELS[var], ov, f"Dice {met['dice']:.3f} | FG {100.0 * float(m.mean()):.1f}% | C {cc}")
        )

    for idx, (head, img, foot) in enumerate(panels):
        ax: Axes = axes.ravel()[idx]
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(head, fontsize=9.2, fontweight="bold")
        ax.set_xlabel(foot, fontsize=8.0)

    legend_handles = [
        Patch(facecolor=TP_COLOR_RGB / 255.0, label="TP"),
        Patch(facecolor=FP_COLOR_RGB / 255.0, label="FP"),
        Patch(facecolor=FN_COLOR_RGB / 255.0, label="FN"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=3, frameon=True, fontsize=8.4)
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=10.6, fontweight="bold")
    return fig


def plot_dice_up_cls_flip_example(
    project_root: str | Path,
    sample_id: str,
    target_variant: str,
    oof_df: pd.DataFrame,
    *,
    family: str = "otsu",
    spec_id: str = "rbf_bal_C3_g001",
) -> Figure:
    """Visualize a case where morphology raises segmentation Dice yet distorts morphology features & OOF SVM."""
    configure_notebook_style()
    root = Path(project_root).expanduser().resolve()
    manifest = load_split_manifest(root, verify=True)
    row = manifest.loc[manifest["sample_id"].eq(sample_id)].iloc[0]
    if row["split"] == "test":
        raise ValueError(f"Forbidden test sample: {sample_id}")

    rgb = load_rgb(root / str(row["image_path"]))
    ref = load_reference_mask(root / str(row["mask_path"]))
    if family == "otsu":
        otsu_cfg = load_otsu_config(root)
        _, raw_m0 = otsu_threshold_mask(rgb, foreground_polarity=str(otsu_cfg["foreground_polarity"]))
    else:
        km = load_kmeans_model(root, k=4, verify_provenance=True)
        lbl = nearest_clusters(rgb_to_lab(rgb).reshape(-1, 3), km["cluster_centers"]).reshape(rgb.shape[:2])
        raw_m0 = np.isin(lbl, km["foreground_clusters"])

    masks = ablation_morphology_variants(raw_m0)
    m0 = masks["M0"]
    mc = masks[target_variant]
    d0 = binary_segmentation_metrics(m0, ref)["dice"]
    dc = binary_segmentation_metrics(mc, ref)["dice"]
    ov0, _ = build_error_overlay(rgb, m0, ref, alpha=0.48)
    ovc, _ = build_error_overlay(rgb, mc, ref, alpha=0.48)

    p0, _ = _morphology_payload(m0)
    pc, _ = _morphology_payload(mc)
    diffs = {k: float(pc[k] - p0[k]) / max(abs(float(p0[k])), 1e-6) for k in MORPHOLOGY_FEATURE_NAMES}
    top_feats = sorted(diffs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]

    oof_sub = oof_df.loc[
        oof_df["sample_id"].eq(sample_id)
        & oof_df["family"].eq(family)
        & oof_df["feature_set"].eq("C")
        & oof_df["spec_id"].eq(spec_id)
    ].set_index("variant")
    if oof_sub.empty:
        oof_sub = oof_df.loc[
            oof_df["sample_id"].eq(sample_id)
            & oof_df["family"].eq(family)
            & oof_df["feature_set"].eq("C")
        ].drop_duplicates(subset=["variant"]).set_index("variant")
    pred_m0 = str(oof_sub.loc["M0", "y_pred"]) if "M0" in oof_sub.index else "-"
    pred_cand = str(oof_sub.loc[target_variant, "y_pred"]) if target_variant in oof_sub.index else "-"

    fig = plt.figure(figsize=(14.6, 4.1), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.35])
    panels = [
        ("RGB + Kontur GT", render_rgb_with_gt_contour(rgb, ref), f"True: {row['category']}"),
        ("M0 (Raw)", ov0, f"Dice {d0:.3f} | Pred: {pred_m0}"),
        (f"{VARIANT_SHORT_LABELS[target_variant]} (ΔDice {dc - d0:+.3f})", ovc, f"Dice {dc:.3f} | Pred: {pred_cand}"),
    ]
    for i, (ttl, img, xlb) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(ttl, fontsize=9.2, fontweight="bold")
        ax.set_xlabel(xlb, fontsize=8.0)

    ax_b = fig.add_subplot(gs[0, 3])
    names = [k for k, _ in reversed(top_feats)]
    vals = [100.0 * np.clip(v, -2.5, 2.5) for _, v in reversed(top_feats)]
    colors = ["#2EA043" if v >= 0 else "#C0392B" for v in vals]
    y_pos = np.arange(len(names))
    ax_b.barh(y_pos, vals, color=colors, height=0.62)
    ax_b.axvline(0.0, color="#2C3E50", linewidth=0.9)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(names, fontsize=7.6)
    ax_b.set_xlabel("Perubahan Relatif vs M0 (%, klip ±250%)", fontsize=8.5)
    ax_b.set_title("8 Fitur Set B Paling Berubah", fontsize=9.2, fontweight="bold")
    ax_b.grid(axis="x", alpha=0.25)
    return fig


def main() -> None:
    """Execute the full Task 2 controlled morphology ablation and print a concise summary."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    cv2.setNumThreads(0)
    root = Path(__file__).resolve().parents[2]
    outputs = run_controlled_morphology_ablation(root, use_cache=False, save_outputs=True)
    print("Saved Task 2 results to:", (root / RESULTS_ABLATION_DIR_REL).as_posix())
    print("Manifest train/val/test:", outputs.manifest["train_images"], outputs.manifest["val_readonly_images"], outputs.manifest["test_images_touched"])


if __name__ == "__main__":
    main()

