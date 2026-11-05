"""Overlap metrics, pixel confusion statistics, and equal-category macro aggregation."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
import pandas as pd


def _coerce_binary_pair(
    predicted: np.ndarray | list,
    reference: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray]:
    pred_arr = np.asarray(predicted)
    ref_arr = np.asarray(reference)

    if pred_arr.ndim == 0 or ref_arr.ndim == 0:
        raise ValueError("Predicted and reference masks must be at least 1D arrays.")
    if pred_arr.shape != ref_arr.shape:
        raise ValueError(
            f"Mask shape mismatch: predicted {pred_arr.shape} vs reference {ref_arr.shape}."
        )

    for name, arr in (("predicted", pred_arr), ("reference", ref_arr)):
        if arr.dtype == np.bool_:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{name} mask must be boolean or binary {0, 1}, got dtype {arr.dtype}.")
        if not np.isfinite(arr).all() or not np.isin(arr, (0, 1)).all():
            raise ValueError(f"{name} mask contains values outside binary {{0, 1}}.")

    return pred_arr.astype(bool, copy=False), ref_arr.astype(bool, copy=False)


def overlap_scores(
    predicted: np.ndarray | list,
    reference: np.ndarray | list,
) -> tuple[float, float]:
    """Compute (Dice, IoU) for equal-shape binary masks; both-empty returns (1.0, 1.0)."""
    pred, ref = _coerce_binary_pair(predicted, reference)

    intersection = int(np.count_nonzero(pred & ref))
    union = int(np.count_nonzero(pred | ref))
    total = int(np.count_nonzero(pred) + np.count_nonzero(ref))

    dice = float(2 * intersection / total) if total > 0 else 1.0
    iou = float(intersection / union) if union > 0 else 1.0
    return dice, iou


def confusion_counts(
    predicted: np.ndarray | list,
    reference: np.ndarray | list,
) -> dict[str, int]:
    """Compute pixel-level TP, FP, FN, and TN counts for equal-shape binary masks."""
    pred, ref = _coerce_binary_pair(predicted, reference)
    return {
        "tp": int(np.count_nonzero(pred & ref)),
        "fp": int(np.count_nonzero(pred & ~ref)),
        "fn": int(np.count_nonzero(~pred & ref)),
        "tn": int(np.count_nonzero(~pred & ~ref)),
    }


def binary_segmentation_metrics(
    predicted: np.ndarray | list,
    reference: np.ndarray | list,
) -> dict[str, float | int]:
    """Compute Dice, IoU, precision, recall, coverage, and TP/FP/FN/TN counts.

    Precision and recall return NaN explicitly when their denominator is zero,
    while preserving the historical both-empty overlap convention (Dice=IoU=1.0).
    """
    pred, ref = _coerce_binary_pair(predicted, reference)
    dice, iou = overlap_scores(pred, ref)
    counts = confusion_counts(pred, ref)

    pred_pos = counts["tp"] + counts["fp"]
    ref_pos = counts["tp"] + counts["fn"]
    precision = float(counts["tp"] / pred_pos) if pred_pos > 0 else math.nan
    recall = float(counts["tp"] / ref_pos) if ref_pos > 0 else math.nan

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "predicted_foreground_fraction": float(pred.mean()) if pred.size > 0 else math.nan,
        "reference_foreground_fraction": float(ref.mean()) if ref.size > 0 else math.nan,
        **counts,
    }


def _validate_results_frame(
    results: pd.DataFrame,
    expected_categories: Iterable[str] | None = None,
) -> list[str]:
    if not isinstance(results, pd.DataFrame) or results.empty:
        raise ValueError("Results table must be a non-empty DataFrame.")

    required = {"method", "category", "dice", "iou"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results table missing required columns: {sorted(missing)}")

    if results["method"].isna().any() or results["category"].isna().any():
        raise ValueError("Method and category columns must not contain missing values.")

    for metric in ("dice", "iou"):
        if not pd.api.types.is_numeric_dtype(results[metric]):
            raise ValueError(f"Metric column '{metric}' must be numeric.")
        values = results[metric].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Metric column '{metric}' contains NaN or infinite values.")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"Metric column '{metric}' contains values outside [0, 1].")

    if "sample_id" in results.columns:
        if results["sample_id"].isna().any():
            raise ValueError("Column 'sample_id' must not contain missing values.")
        if results.duplicated(["method", "sample_id"]).any():
            raise ValueError("Duplicate (method, sample_id) entries detected in results.")

        sample_sets = {
            method: set(group["sample_id"])
            for method, group in results.groupby("method", observed=True)
        }
        iterator = iter(sample_sets.items())
        first_method, baseline_ids = next(iterator)
        for method, ids in iterator:
            if ids != baseline_ids:
                raise ValueError(
                    f"Cohort sample_id mismatch between '{first_method}' and '{method}'."
                )

        if results.groupby("sample_id", observed=True)["category"].nunique().gt(1).any():
            raise ValueError("Inconsistent category assignment across methods for the same sample_id.")
        if "group_id" in results.columns:
            if results["group_id"].isna().any():
                raise ValueError("Column 'group_id' must not contain missing values.")
            if results.groupby("sample_id", observed=True)["group_id"].nunique().gt(1).any():
                raise ValueError(
                    "Inconsistent group_id assignment across methods for the same sample_id."
                )

    observed_categories = sorted(results["category"].unique())
    expected_set = set(expected_categories) if expected_categories is not None else set(observed_categories)
    if not expected_set:
        raise ValueError("Expected category set must be non-empty.")

    for method, group in results.groupby("method", observed=True):
        method_cats = set(group["category"].unique())
        missing_cats = sorted(expected_set - method_cats)
        extra_cats = sorted(method_cats - expected_set)
        if missing_cats or extra_cats:
            raise ValueError(
                f"Category mismatch for method '{method}': "
                f"missing={missing_cats}, unexpected={extra_cats}."
            )

    return sorted(expected_set)


def category_score_table(
    results: pd.DataFrame,
    expected_categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute within-category mean image Dice and IoU indexed by (method, category)."""
    categories = _validate_results_frame(results, expected_categories=expected_categories)
    grouped = (
        results.groupby(["method", "category"], observed=True)[["dice", "iou"]]
        .mean()
    )
    methods = list(dict.fromkeys(results["method"]))
    index = pd.MultiIndex.from_product([methods, categories], names=["method", "category"])
    return grouped.reindex(index)


def category_macro_scores(
    results: pd.DataFrame,
    expected_categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute within-category mean image score first, then equal-weight mean across categories."""
    per_category = category_score_table(results, expected_categories=expected_categories)
    macro = (
        per_category.groupby("method", observed=True)[["dice", "iou"]]
        .mean()
        .rename(columns={"dice": "mean_category_dice", "iou": "mean_category_iou"})
    )
    return macro
