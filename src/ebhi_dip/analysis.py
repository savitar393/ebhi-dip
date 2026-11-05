"""Validation score harmonization, paired method comparison, case ranking, and case building."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ebhi_dip.dataset import (
    EXPECTED_CATEGORIES,
    REFERENCE_THRESHOLD,
    load_raw_mask,
    load_rgb,
    load_split_manifest,
)
from ebhi_dip.metrics import binary_segmentation_metrics, category_macro_scores
from ebhi_dip.paths import resolve_manifest_path, verify_frozen_split
from ebhi_dip.provenance import sha256_bytes, verify_helper_provenance
from ebhi_dip.segmentation import (
    STAGES,
    load_kmeans_model,
    load_otsu_config,
    otsu_threshold_mask,
    predict_mask,
)

TIE_TOLERANCE: float = 1e-12
PARITY_ATOL: float = 1e-10

RANKING_MODES: tuple[str, ...] = (
    "worst_a",
    "worst_b",
    "a_advantage",
    "b_advantage",
    "both_poor",
    "best_a",
    "best_b",
    "ties",
)


def method_label(method_id: str) -> str:
    """Format canonical method IDs into concise human-readable labels."""
    if method_id == "all_foreground":
        return "All foreground"
    if "_" not in method_id:
        return method_id
    family, stage = method_id.split("_", 1)
    if family == "otsu":
        return f"Otsu {stage}"
    if family.startswith("k") and family[1:].isdigit():
        return f"K-means k={family[1:]}, {stage}"
    return method_id


def selection_ranking_key(summary: pd.DataFrame, method_id: str) -> tuple[float, float, int, int]:
    """Return established selection key: 4-dec Dice, 4-dec IoU, fewer stages, fewer clusters."""
    row = summary.loc[method_id]
    stage = int(method_id.rsplit("M", 1)[1]) if "_M" in method_id else 99
    k_val = int(method_id[1]) if method_id.startswith("k") and method_id[1:2].isdigit() else 0
    return (
        round(float(row["mean_category_dice"]), 4),
        round(float(row["mean_category_iou"]), 4),
        -stage,
        -k_val,
    )


def _validate_cohort_alignment(table: pd.DataFrame, validation_cohort: pd.DataFrame, label: str) -> None:
    if not table["sample_id"].is_unique:
        raise ValueError(f"Duplicate sample_id values in {label}.")
    if set(table["sample_id"]) != set(validation_cohort["sample_id"]):
        raise ValueError(f"Sample cohort mismatch in {label} relative to frozen validation split.")

    joined = table.merge(
        validation_cohort[["sample_id", "category", "group_id"]],
        on="sample_id",
        how="left",
        suffixes=("", "_frozen"),
        validate="one_to_one",
    )
    for col in ("category", "group_id"):
        if not joined[col].eq(joined[f"{col}_frozen"]).all():
            raise ValueError(f"Metadata mismatch in column '{col}' for {label}.")


def load_validation_results(project_root: str | Path) -> pd.DataFrame:
    """Load, validate, reconcile, and harmonize the 13 validation methods across 372 images."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    split_df = load_split_manifest(root, verify=False)
    validation = split_df.loc[split_df["split"].eq("val")].copy()

    otsu_path = root / "results/otsu_morph_val.csv"
    kmeans_path = root / "results/kmeans_morph_val.csv"
    run_path = root / "results/kmeans_morph_val_run.json"
    for req_path in (otsu_path, kmeans_path, run_path):
        if not req_path.is_file():
            raise FileNotFoundError(f"Missing validation artifact: {req_path}")

    otsu = pd.read_csv(otsu_path, encoding="utf-8-sig")
    kmeans = pd.read_csv(kmeans_path, encoding="utf-8-sig")
    run_meta = json.loads(run_path.read_text(encoding="utf-8"))
    otsu_config = load_otsu_config(root)

    if run_meta.get("evaluation_subset") != "val" or run_meta.get("split_sha256") != split_sha:
        raise ValueError("Run metadata subset or split SHA mismatch.")
    if run_meta.get("image_count") != len(validation) or run_meta.get("result_row_count") != len(kmeans):
        raise ValueError("Run metadata row/image count mismatch.")
    if run_meta.get("reference_threshold") != REFERENCE_THRESHOLD:
        raise ValueError("Run metadata reference threshold mismatch.")

    verify_helper_provenance(root, "kmeans_prediction", str(run_meta["prediction_helper_sha256"]))
    verify_helper_provenance(root, "postprocessing", str(run_meta["postprocessing_helper_sha256"]))

    for k in (2, 3, 4):
        model_path = root / "models" / f"kmeans_lab_k{k}_mapped.json"
        if sha256_bytes(model_path) != run_meta["model_sha256"][f"k{k}"]:
            raise ValueError(f"Mapped model SHA-256 mismatch for k={k} relative to run metadata.")
        load_kmeans_model(root, k=k, verify_provenance=True)

    if not otsu["foreground_polarity"].eq(otsu_config["foreground_polarity"]).all():
        raise ValueError("Otsu validation CSV foreground polarity differs from configs/otsu_raw.json.")
    if not kmeans["split"].eq("val").all():
        raise ValueError("K-means validation CSV contains non-validation rows.")

    expected_km_methods = {f"k{k}_{st}" for k in (2, 3, 4) for st in STAGES} | {"all_foreground"}
    if set(kmeans["method"].unique()) != expected_km_methods:
        raise ValueError("Unexpected method set in results/kmeans_morph_val.csv.")
    if kmeans.duplicated(["sample_id", "method"]).any():
        raise ValueError("Duplicate (sample_id, method) rows in results/kmeans_morph_val.csv.")

    _validate_cohort_alignment(otsu, validation, "results/otsu_morph_val.csv")
    for method_name, method_table in kmeans.groupby("method", observed=True):
        _validate_cohort_alignment(method_table, validation, f"kmeans:{method_name}")

    km_control = kmeans.loc[kmeans["method"].eq("all_foreground")].set_index("sample_id").sort_index()
    otsu_indexed = otsu.set_index("sample_id").sort_index()
    for metric in ("dice", "iou"):
        np.testing.assert_allclose(
            km_control[metric].to_numpy(dtype=np.float64),
            otsu_indexed[f"all_foreground_{metric}"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )

    ref_fraction = km_control["reference_foreground_fraction"].to_numpy(dtype=np.float64)
    if not np.isfinite(ref_fraction).all() or np.any((ref_fraction < 0.0) | (ref_fraction > 1.0)):
        raise ValueError("Invalid reference_foreground_fraction values in K-means validation CSV.")
    np.testing.assert_allclose(
        km_control["dice"].to_numpy(dtype=np.float64),
        2.0 * ref_fraction / (1.0 + ref_fraction),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        km_control["iou"].to_numpy(dtype=np.float64),
        ref_fraction,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        km_control["predicted_foreground_fraction"].to_numpy(dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    if not kmeans.groupby("sample_id", observed=True)["reference_foreground_fraction"].nunique().eq(1).all():
        raise ValueError("Inconsistent reference_foreground_fraction across methods in K-means CSV.")

    otsu_thr_map = otsu_indexed["otsu_threshold"].astype(float)
    ref_frac_map = km_control["reference_foreground_fraction"].astype(float)

    km_part = kmeans[
        [
            "sample_id",
            "category",
            "group_id",
            "split",
            "method",
            "dice",
            "iou",
            "predicted_foreground_fraction",
            "reference_foreground_fraction",
        ]
    ].copy()
    km_part["otsu_threshold"] = km_part["sample_id"].map(otsu_thr_map)

    pieces = [km_part]
    for stage in STAGES:
        part = otsu[
            ["sample_id", "category", "group_id", "otsu_threshold", f"{stage}_dice", f"{stage}_iou"]
        ].rename(columns={f"{stage}_dice": "dice", f"{stage}_iou": "iou"}).copy()
        part["split"] = "val"
        part["method"] = f"otsu_{stage}"
        part["predicted_foreground_fraction"] = np.nan
        part["reference_foreground_fraction"] = part["sample_id"].map(ref_frac_map)
        pieces.append(
            part[
                [
                    "sample_id",
                    "category",
                    "group_id",
                    "split",
                    "method",
                    "dice",
                    "iou",
                    "predicted_foreground_fraction",
                    "reference_foreground_fraction",
                    "otsu_threshold",
                ]
            ]
        )

    harmonized = pd.concat(pieces, ignore_index=True)
    if len(harmonized) != len(validation) * 13 or harmonized["method"].nunique() != 13:
        raise ValueError("Harmonized validation table does not have 13 methods x validation cohort rows.")
    if harmonized.duplicated(["sample_id", "method"]).any():
        raise ValueError("Harmonized validation table contains duplicate (sample_id, method) rows.")

    metric_vals = harmonized[["dice", "iou"]].to_numpy(dtype=np.float64)
    if not np.isfinite(metric_vals).all() or np.any((metric_vals < 0.0) | (metric_vals > 1.0)):
        raise ValueError("Harmonized validation metrics contain non-finite or out-of-range values.")

    return harmonized


def summarize_validation_methods(
    results: pd.DataFrame,
    expected_categories: Iterable[str] | None = EXPECTED_CATEGORIES,
) -> pd.DataFrame:
    """Compute equal-category macro summary and deltas versus all_foreground."""
    summary = category_macro_scores(results, expected_categories=expected_categories).copy()
    if "all_foreground" in summary.index:
        summary["delta_dice_vs_control"] = (
            summary["mean_category_dice"] - float(summary.loc["all_foreground", "mean_category_dice"])
        )
        summary["delta_iou_vs_control"] = (
            summary["mean_category_iou"] - float(summary.loc["all_foreground", "mean_category_iou"])
        )
    return summary.sort_values(["mean_category_dice", "mean_category_iou"], ascending=False)


def pair_scores(
    results: pd.DataFrame,
    method_a: str,
    method_b: str,
    category: str | None = None,
    tie_tolerance: float = TIE_TOLERANCE,
) -> pd.DataFrame:
    """Join per-image validation scores one-to-one for method_a vs method_b."""
    if not isinstance(results, pd.DataFrame) or results.empty:
        raise ValueError("Results DataFrame must be non-empty.")
    if tie_tolerance < 0 or not np.isfinite(tie_tolerance):
        raise ValueError(f"tie_tolerance must be finite and non-negative, got {tie_tolerance}.")

    required = {"sample_id", "category", "group_id", "method", "dice", "iou"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results missing required columns for pairing: {sorted(missing)}")

    if "split" in results.columns and not results["split"].eq("val").all():
        raise ValueError("Paired case analysis only permits validation ('val') split records.")

    available_methods = set(results["method"].unique())
    for m_name in (method_a, method_b):
        if m_name not in available_methods:
            raise ValueError(f"Method '{m_name}' not found in results. Available: {sorted(available_methods)}")

    cols = ["sample_id", "category", "group_id", "dice", "iou"]
    if "reference_foreground_fraction" in results.columns:
        cols.append("reference_foreground_fraction")

    sub_a = results.loc[results["method"].eq(method_a), cols].copy()
    sub_b = results.loc[results["method"].eq(method_b), cols].copy()

    if sub_a["sample_id"].isna().any() or sub_b["sample_id"].isna().any():
        raise ValueError("Missing sample_id values in paired inputs.")
    if not sub_a["sample_id"].is_unique:
        raise ValueError(f"Duplicate sample_id entries for method_a='{method_a}'.")
    if not sub_b["sample_id"].is_unique:
        raise ValueError(f"Duplicate sample_id entries for method_b='{method_b}'.")

    ids_a = set(sub_a["sample_id"])
    ids_b = set(sub_b["sample_id"])
    if ids_a != ids_b or not ids_a:
        only_a = sorted(ids_a - ids_b)[:5]
        only_b = sorted(ids_b - ids_a)[:5]
        raise ValueError(
            f"Unpaired sample_id mismatch between '{method_a}' and '{method_b}': "
            f"only_in_a={only_a}, only_in_b={only_b}."
        )

    merged = sub_a.merge(sub_b, on="sample_id", how="inner", suffixes=("_a", "_b"), validate="one_to_one")
    for meta_col in ("category", "group_id"):
        if merged[f"{meta_col}_a"].isna().any() or merged[f"{meta_col}_b"].isna().any():
            raise ValueError(f"Missing '{meta_col}' metadata in paired inputs.")
        if not merged[f"{meta_col}_a"].eq(merged[f"{meta_col}_b"]).all():
            raise ValueError(f"Cohort '{meta_col}' mismatch between '{method_a}' and '{method_b}'.")

    for metric_col in ("dice_a", "iou_a", "dice_b", "iou_b"):
        vals = merged[metric_col].to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all() or np.any((vals < 0.0) | (vals > 1.0)):
            raise ValueError(f"Invalid metric values in '{metric_col}'.")

    paired = pd.DataFrame(
        {
            "sample_id": merged["sample_id"],
            "category": merged["category_a"],
            "group_id": merged["group_id_a"],
            "method_a": method_a,
            "method_b": method_b,
            "dice_a": merged["dice_a"].astype(np.float64),
            "iou_a": merged["iou_a"].astype(np.float64),
            "dice_b": merged["dice_b"].astype(np.float64),
            "iou_b": merged["iou_b"].astype(np.float64),
        }
    )
    if "reference_foreground_fraction_a" in merged.columns:
        paired["reference_foreground_fraction"] = merged["reference_foreground_fraction_a"].astype(
            np.float64
        )

    if category is not None and category != "All":
        known_categories = set(paired["category"].unique())
        if category not in known_categories:
            raise ValueError(
                f"Unknown category filter '{category}'. Known categories: {sorted(known_categories)}"
            )
        paired = paired.loc[paired["category"].eq(category)].copy()

    delta_dice = paired["dice_a"] - paired["dice_b"]
    delta_iou = paired["iou_a"] - paired["iou_b"]
    paired["delta_dice"] = delta_dice
    paired["delta_iou"] = delta_iou
    paired["max_dice"] = np.maximum(paired["dice_a"], paired["dice_b"])
    paired["min_dice"] = np.minimum(paired["dice_a"], paired["dice_b"])

    outcomes = np.where(
        delta_dice > tie_tolerance,
        "a_higher",
        np.where(delta_dice < -tie_tolerance, "b_higher", "tie"),
    )
    paired["comparison_outcome"] = outcomes
    return paired.sort_values("sample_id", ascending=True, kind="mergesort").reset_index(drop=True)


def paired_outcome_counts(
    paired: pd.DataFrame,
    expected_categories: Iterable[str] | None = None,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Compute per-image outcome counts (A higher, B higher, tie) overall and per category."""
    overall = {
        "a_higher": int(paired["comparison_outcome"].eq("a_higher").sum()),
        "b_higher": int(paired["comparison_outcome"].eq("b_higher").sum()),
        "tie": int(paired["comparison_outcome"].eq("tie").sum()),
        "total": int(len(paired)),
    }

    categories = (
        list(expected_categories)
        if expected_categories is not None
        else sorted(paired["category"].unique())
    )
    rows = []
    for cat in categories:
        sub = paired.loc[paired["category"].eq(cat)]
        rows.append(
            {
                "category": cat,
                "a_higher": int(sub["comparison_outcome"].eq("a_higher").sum()),
                "b_higher": int(sub["comparison_outcome"].eq("b_higher").sum()),
                "tie": int(sub["comparison_outcome"].eq("tie").sum()),
                "total": int(len(sub)),
            }
        )
    by_category = pd.DataFrame(rows).set_index("category")
    return overall, by_category


def mode_description(mode: str, method_a: str, method_b: str) -> str:
    """Return concise Indonesian label distinguishing single-method worst from pairwise advantage."""
    labels = {
        "worst_a": f"Worst {method_a} (Dice A terendah)",
        "worst_b": f"Worst {method_b} (Dice B terendah)",
        "a_advantage": f"Keunggulan terbesar {method_a} (ΔDice > 0 terbesar)",
        "b_advantage": f"Keunggulan terbesar {method_b} (ΔDice < 0 terkecil)",
        "both_poor": "Kedua metode lemah (max(Dice A, Dice B) terendah)",
        "best_a": f"Best {method_a} (Dice A tertinggi)",
        "best_b": f"Best {method_b} (Dice B tertinggi)",
        "ties": "Tie (|ΔDice| <= 1e-12)",
    }
    if mode not in labels:
        raise ValueError(f"Unknown ranking mode '{mode}'. Supported: {RANKING_MODES}")
    return labels[mode]


def rank_cases(
    paired: pd.DataFrame,
    mode: str,
    n: int = 3,
    tie_tolerance: float = TIE_TOLERANCE,
) -> pd.DataFrame:
    """Rank paired validation cases deterministically with sample_id ascending tie-break."""
    if mode not in RANKING_MODES:
        raise ValueError(f"Unsupported ranking mode '{mode}'. Supported: {RANKING_MODES}")
    if int(n) < 1:
        raise ValueError(f"Rank count n must be >= 1, got {n}.")

    base_cols = list(paired.columns)
    if paired.empty:
        empty = paired.copy()
        empty["selection_mode"] = pd.Series(dtype=object)
        empty["selection_reason"] = pd.Series(dtype=object)
        return empty

    method_a = str(paired["method_a"].iloc[0])
    method_b = str(paired["method_b"].iloc[0])
    reason = mode_description(mode, method_a, method_b)

    work = paired.copy()
    if mode == "worst_a":
        ranked = work.sort_values(["dice_a", "sample_id"], ascending=[True, True], kind="mergesort")
    elif mode == "worst_b":
        ranked = work.sort_values(["dice_b", "sample_id"], ascending=[True, True], kind="mergesort")
    elif mode == "a_advantage":
        filtered = work.loc[work["delta_dice"] > tie_tolerance].copy()
        ranked = filtered.sort_values(
            ["delta_dice", "sample_id"], ascending=[False, True], kind="mergesort"
        )
    elif mode == "b_advantage":
        filtered = work.loc[work["delta_dice"] < -tie_tolerance].copy()
        ranked = filtered.sort_values(
            ["delta_dice", "sample_id"], ascending=[True, True], kind="mergesort"
        )
    elif mode == "both_poor":
        ranked = work.sort_values(
            ["max_dice", "min_dice", "sample_id"],
            ascending=[True, True, True],
            kind="mergesort",
        )
    elif mode == "best_a":
        ranked = work.sort_values(["dice_a", "sample_id"], ascending=[False, True], kind="mergesort")
    elif mode == "best_b":
        ranked = work.sort_values(["dice_b", "sample_id"], ascending=[False, True], kind="mergesort")
    elif mode == "ties":
        filtered = work.loc[work["delta_dice"].abs() <= tie_tolerance].copy()
        filtered["_abs_delta"] = filtered["delta_dice"].abs()
        ranked = filtered.sort_values(
            ["_abs_delta", "sample_id"], ascending=[True, True], kind="mergesort"
        ).drop(columns=["_abs_delta"])

    selected = ranked[base_cols].head(int(n)).copy().reset_index(drop=True)
    selected["selection_mode"] = mode
    selected["selection_reason"] = reason
    return selected


def select_distinct_cases(
    paired: pd.DataFrame,
    mode_specs: Sequence[tuple[str, int]] = (
        ("worst_a", 1),
        ("worst_b", 1),
        ("a_advantage", 1),
        ("b_advantage", 1),
    ),
    tie_tolerance: float = TIE_TOLERANCE,
) -> pd.DataFrame:
    """Select cases across ranking modes, deduplicating by sample_id while preserving all reasons."""
    records_by_id: dict[str, dict[str, object]] = {}
    modes_by_id: dict[str, list[str]] = {}
    reasons_by_id: dict[str, list[str]] = {}

    for mode, count in mode_specs:
        ranked = rank_cases(paired, mode=mode, n=count, tie_tolerance=tie_tolerance)
        for _, row in ranked.iterrows():
            sid = str(row["sample_id"])
            if sid not in records_by_id:
                records_by_id[sid] = row.drop(labels=["selection_mode", "selection_reason"]).to_dict()
                modes_by_id[sid] = []
                reasons_by_id[sid] = []
            if mode not in modes_by_id[sid]:
                modes_by_id[sid].append(mode)
                reasons_by_id[sid].append(str(row["selection_reason"]))

    if not records_by_id:
        empty = paired.iloc[0:0].copy()
        empty["selection_modes"] = pd.Series(dtype=object)
        empty["selection_reasons"] = pd.Series(dtype=object)
        return empty

    rows = []
    for sid, base in records_by_id.items():
        entry = dict(base)
        entry["selection_modes"] = tuple(modes_by_id[sid])
        entry["selection_reasons"] = "; ".join(reasons_by_id[sid])
        rows.append(entry)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class SegmentationCase:
    """Complete post-hoc case container for paired visual and histogram inspection."""

    sample_id: str
    category: str
    group_id: str
    split: str
    method_a: str
    method_b: str
    selection_reasons: tuple[str, ...]
    image_rgb: np.ndarray
    raw_reference_l: np.ndarray
    binary_reference: np.ndarray
    prediction_a: np.ndarray
    prediction_b: np.ndarray
    metrics_a: dict[str, float | int]
    metrics_b: dict[str, float | int]
    saved_dice_a: float
    saved_iou_a: float
    saved_dice_b: float
    saved_iou_b: float
    delta_dice: float
    delta_iou: float
    otsu_threshold: float
    otsu_polarity: str
    parity_atol: float = PARITY_ATOL


def build_case(
    project_root: str | Path,
    row: pd.Series | dict[str, object],
    method_a: str,
    method_b: str,
    selection_reasons: Sequence[str] | str = (),
    data_root_override: str | Path | None = None,
    split_df: pd.DataFrame | None = None,
    atol: float = PARITY_ATOL,
) -> SegmentationCase:
    """Resolve raw validation image/mask, run fixed full-image inference, and verify CSV parity."""
    root = Path(project_root).expanduser().resolve()
    manifest = split_df if split_df is not None else load_split_manifest(root, verify=True)

    sample_id = str(row["sample_id"])
    matches = manifest.loc[manifest["sample_id"].eq(sample_id)]
    if len(matches) != 1:
        raise KeyError(f"Sample '{sample_id}' not found uniquely in frozen split manifest.")

    meta = matches.iloc[0]
    subset = str(meta["split"])
    if subset != "val":
        raise ValueError(
            f"Case builder strictly requires validation subset ('val'), got '{subset}' for '{sample_id}'."
        )
    if str(row["category"]) != str(meta["category"]) or str(row["group_id"]) != str(meta["group_id"]):
        raise ValueError(f"Row metadata mismatch against frozen split for '{sample_id}'.")

    image_path = resolve_manifest_path(root, str(meta["image_path"]), data_root_override=data_root_override)
    mask_path = resolve_manifest_path(root, str(meta["mask_path"]), data_root_override=data_root_override)

    # Fixed inference uses ONLY original RGB and saved train-derived artifacts (all 50,176 pixels).
    image_rgb = load_rgb(image_path)
    prediction_a = predict_mask(image_rgb, method_a, root)
    prediction_b = predict_mask(image_rgb, method_b, root)

    otsu_cfg = load_otsu_config(root)
    otsu_polarity = str(otsu_cfg["foreground_polarity"])
    otsu_thr, _ = otsu_threshold_mask(image_rgb, foreground_polarity=otsu_polarity)

    # Reference mask is loaded strictly AFTER prediction for post-hoc evaluation and overlays.
    raw_reference_l = load_raw_mask(mask_path)
    if raw_reference_l.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"Image/reference dimension mismatch for '{sample_id}': "
            f"{image_rgb.shape[:2]} vs {raw_reference_l.shape}."
        )
    binary_reference = raw_reference_l >= REFERENCE_THRESHOLD

    metrics_a = binary_segmentation_metrics(prediction_a, binary_reference)
    metrics_b = binary_segmentation_metrics(prediction_b, binary_reference)

    saved_dice_a = float(row["dice_a"])
    saved_iou_a = float(row["iou_a"])
    saved_dice_b = float(row["dice_b"])
    saved_iou_b = float(row["iou_b"])

    checks = (
        (method_a, "dice", float(metrics_a["dice"]), saved_dice_a),
        (method_a, "iou", float(metrics_a["iou"]), saved_iou_a),
        (method_b, "dice", float(metrics_b["dice"]), saved_dice_b),
        (method_b, "iou", float(metrics_b["iou"]), saved_iou_b),
    )
    for m_id, metric_name, recomputed, saved in checks:
        diff = abs(recomputed - saved)
        if not np.isfinite(diff) or diff > atol:
            raise ValueError(
                f"CSV parity check failed for sample '{sample_id}' ({m_id} {metric_name}): "
                f"recomputed={recomputed:.15f}, saved={saved:.15f}, |diff|={diff:.3e} > atol={atol:.1e}."
            )

    if isinstance(selection_reasons, str):
        reasons_tuple = tuple(part.strip() for part in selection_reasons.split(";") if part.strip())
    else:
        reasons_tuple = tuple(str(item) for item in selection_reasons if str(item).strip())

    return SegmentationCase(
        sample_id=sample_id,
        category=str(meta["category"]),
        group_id=str(meta["group_id"]),
        split=subset,
        method_a=method_a,
        method_b=method_b,
        selection_reasons=reasons_tuple,
        image_rgb=image_rgb,
        raw_reference_l=raw_reference_l,
        binary_reference=binary_reference,
        prediction_a=prediction_a,
        prediction_b=prediction_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        saved_dice_a=saved_dice_a,
        saved_iou_a=saved_iou_a,
        saved_dice_b=saved_dice_b,
        saved_iou_b=saved_iou_b,
        delta_dice=float(metrics_a["dice"]) - float(metrics_b["dice"]),
        delta_iou=float(metrics_a["iou"]) - float(metrics_b["iou"]),
        otsu_threshold=otsu_thr,
        otsu_polarity=otsu_polarity,
        parity_atol=atol,
    )
