"""Diagnostic and visual analysis helpers for morphological postprocessing (M0, M1, M2)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ebhi_dip.analysis import load_validation_results, method_label
from ebhi_dip.dataset import (
    EXPECTED_CATEGORIES,
    load_rgb,
    load_reference_mask,
    load_split_manifest,
)
from ebhi_dip.metrics import binary_segmentation_metrics
from ebhi_dip.paths import verify_frozen_split
from ebhi_dip.plotting import (
    FN_COLOR_RGB,
    FP_COLOR_RGB,
    TP_COLOR_RGB,
    build_error_overlay,
    configure_notebook_style,
)
from ebhi_dip.segmentation import (
    load_kmeans_model,
    load_otsu_config,
    morphology_variants,
    nearest_clusters,
    otsu_threshold_mask,
    rgb_to_lab,
)

MIN_COMPONENT_AREA = 16
NEGLIGIBLE_DELTA_THRESHOLD = 0.005
CROSS_3X3_KERNEL = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

DIAGNOSTIC_FAMILIES: tuple[str, ...] = ("otsu", "k4")
DIAGNOSTIC_STAGES: tuple[str, ...] = ("M0", "M1", "M2")
DEV_SPLITS: tuple[str, ...] = ("train", "val")

AREA_BIN_EDGES: tuple[int, ...] = (1, 2, 5, 16, 32, 128, 1024, 10**9)
AREA_BIN_LABELS: tuple[str, ...] = (
    "1",
    "2–4",
    "5–15",
    "16–31",
    "32–127",
    "128–1023",
    "≥1024",
)

# Consistent visual palettes across image panels and metric charts.
MASK_BG_RGB = np.array([244, 246, 249], dtype=np.uint8)
MASK_KEPT_RGB = np.array([35, 49, 66], dtype=np.uint8)
MASK_SMALL_RGB = np.array([220, 53, 69], dtype=np.uint8)

DIFF_REMOVED_M1_RGB = np.array([220, 53, 69], dtype=np.uint8)
DIFF_REMOVED_M2_RGB = np.array([31, 119, 180], dtype=np.uint8)
DIFF_ADDED_M2_RGB = np.array([243, 156, 18], dtype=np.uint8)
GT_CONTOUR_RGB = np.array([0, 230, 200], dtype=np.uint8)

FAMILY_COLORS: dict[str, str] = {"otsu": "#1F77B4", "k4": "#D95F02"}
STAGE_COLORS: dict[str, str] = {"M0": "#6C757D", "M1": "#2E86AB", "M2": "#D95F02"}
OUTCOME_LABELS: tuple[str, ...] = (
    "Improve (≥ +0.005)",
    "Negligible (|Δ| < 0.005)",
    "Regress (≤ −0.005)",
)
OUTCOME_COLORS: dict[str, str] = {
    OUTCOME_LABELS[0]: "#2EA043",
    OUTCOME_LABELS[1]: "#BDC3C7",
    OUTCOME_LABELS[2]: "#D62728",
}


def file_sha256_map(
    project_root: str | Path,
    relative_paths: Iterable[str | Path],
) -> dict[str, str]:
    """Compute SHA-256 checksums for read-only verification of repository inputs."""
    root = Path(project_root).expanduser().resolve()
    digests: dict[str, str] = {}
    for rel in relative_paths:
        rel_str = Path(rel).as_posix()
        path = root / rel_str
        if not path.is_file():
            raise FileNotFoundError(f"Expected file for integrity check: {path}")
        digests[rel_str] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _style_image_axis(ax: Axes) -> None:
    """Hide image ticks and frames while keeping row/column labels intact."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _predict_family_states(
    image_rgb: np.ndarray,
    otsu_polarity: str,
    km_centers: np.ndarray,
    km_fg_clusters: np.ndarray,
) -> tuple[float, dict[str, dict[str, np.ndarray]]]:
    """Compute Otsu and K-Means k=4 raw, M1, intermediate OPEN, and M2 masks."""
    otsu_thr, otsu_m0 = otsu_threshold_mask(image_rgb, foreground_polarity=otsu_polarity)
    lab = rgb_to_lab(image_rgb)
    km_labels = nearest_clusters(lab.reshape(-1, 3), km_centers).reshape(image_rgb.shape[:2])
    k4_m0 = np.isin(km_labels, km_fg_clusters)

    states: dict[str, dict[str, np.ndarray]] = {}
    for family, raw_mask in (("otsu", otsu_m0), ("k4", k4_m0)):
        stages = morphology_variants(raw_mask)
        opened = (
            cv2.morphologyEx(
                stages["M1"].astype(np.uint8),
                cv2.MORPH_OPEN,
                CROSS_3X3_KERNEL,
                iterations=1,
                borderType=cv2.BORDER_REPLICATE,
            )
            > 0
        )
        states[family] = {
            "M0": stages["M0"],
            "M1": stages["M1"],
            "open": opened,
            "M2": stages["M2"],
        }
    return float(otsu_thr), states


def connected_component_areas(mask: np.ndarray) -> np.ndarray:
    """Return 8-connected foreground component pixel areas sorted ascending."""
    binary = np.asarray(mask, dtype=bool)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    if n_labels <= 1:
        return np.empty(0, dtype=np.int32)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32, copy=False)
    areas.sort()
    return areas


def render_component_highlight_rgb(
    mask: np.ndarray,
    small_threshold: int = MIN_COMPONENT_AREA,
) -> tuple[np.ndarray, int, int]:
    """Render binary mask highlighting components smaller than small_threshold in coral-red."""
    binary = np.asarray(mask, dtype=bool)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )
    canvas = np.broadcast_to(MASK_BG_RGB, (*binary.shape, 3)).copy()
    if n_labels <= 1:
        return canvas, 0, 0

    areas = stats[:, cv2.CC_STAT_AREA]
    fg_areas = areas[1:]
    small_lookup = np.zeros(n_labels, dtype=bool)
    small_lookup[1:] = fg_areas < small_threshold
    kept_lookup = np.zeros(n_labels, dtype=bool)
    kept_lookup[1:] = fg_areas >= small_threshold

    canvas[kept_lookup[labels]] = MASK_KEPT_RGB
    canvas[small_lookup[labels]] = MASK_SMALL_RGB
    return canvas, int(len(fg_areas)), int(np.sum(fg_areas < small_threshold))


def render_transition_diff_rgb(
    m0: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Render disjoint M0 -> M1 -> M2 foreground pixel transitions on a fixed RGB palette."""
    m0_b = np.asarray(m0, dtype=bool)
    m1_b = np.asarray(m1, dtype=bool)
    m2_b = np.asarray(m2, dtype=bool)

    unchanged = m0_b & m1_b & m2_b
    removed_m1 = m0_b & (~m1_b) & (~m2_b)
    removed_m2 = m1_b & (~m2_b)
    added_m2 = m2_b & (~m1_b)

    canvas = np.broadcast_to(MASK_BG_RGB, (*m0_b.shape, 3)).copy()
    canvas[unchanged] = MASK_KEPT_RGB
    canvas[removed_m1] = DIFF_REMOVED_M1_RGB
    canvas[removed_m2] = DIFF_REMOVED_M2_RGB
    canvas[added_m2] = DIFF_ADDED_M2_RGB

    counts = {
        "unchanged_px": int(np.sum(unchanged)),
        "removed_m1_px": int(np.sum(removed_m1)),
        "removed_m2_px": int(np.sum(removed_m2)),
        "added_m2_px": int(np.sum(added_m2)),
    }
    return canvas, counts


def render_rgb_with_gt_contour(
    image_rgb: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    """Draw the ground-truth reference mask boundary cleanly on top of RGB."""
    out = np.ascontiguousarray(image_rgb).copy()
    ref_u8 = np.asarray(reference_mask, dtype=np.uint8)
    contours, _ = cv2.findContours(ref_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(
            out,
            contours,
            -1,
            color=tuple(int(v) for v in GT_CONTOUR_RGB),
            thickness=1,
        )
    return out


def load_development_morphology_csv(project_root: str | Path) -> pd.DataFrame:
    """Load saved 2x3 morphology diagnostics restricted strictly to train+val splits."""
    root = Path(project_root).expanduser().resolve()
    csv_path = root / "results" / "features" / "morphology_diagnostics_2x3.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing saved morphology diagnostics: {csv_path}")

    df = pd.read_csv(csv_path)
    dev = df.loc[df["split"].isin(DEV_SPLITS)].copy()
    if dev["split"].eq("test").any() or dev.empty:
        raise ValueError("Invalid split filtering in saved morphology diagnostics.")

    family_stage = dev["method"].str.split("_", n=1, expand=True)
    dev["family"] = family_stage[0]
    dev["stage"] = family_stage[1]
    dev["fg_pct"] = 100.0 * dev["foreground_pixels"] / dev["image_pixels"]
    return dev.reset_index(drop=True)


def evaluate_val_morphology_cohort(
    project_root: str | Path,
    split: str = "val",
    verify_against_saved: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Evaluate M0, M1, OPEN, and M2 transitions, pixel flips, and component areas on dev data."""
    if split not in DEV_SPLITS:
        raise ValueError(f"Forbidden or unknown split '{split}'; expected one of {DEV_SPLITS}.")

    root = Path(project_root).expanduser().resolve()
    verify_frozen_split(root)
    manifest = load_split_manifest(root, verify=True)
    cohort = manifest.loc[manifest["split"].eq(split)].reset_index(drop=True)

    otsu_cfg = load_otsu_config(root)
    otsu_polarity = str(otsu_cfg["foreground_polarity"])
    km_model = load_kmeans_model(root, k=4, verify_provenance=True)
    km_centers = np.asarray(km_model["cluster_centers"], dtype=np.float64)
    km_fg_clusters = np.asarray(km_model["foreground_clusters"], dtype=np.int64)

    rows: list[dict[str, object]] = []
    area_chunks: dict[tuple[str, str], list[np.ndarray]] = {
        (fam, stage): [] for fam in DIAGNOSTIC_FAMILIES for stage in DIAGNOSTIC_STAGES
    }

    for rec in cohort.itertuples(index=False):
        rgb = load_rgb(root / rec.image_path)
        ref = load_reference_mask(root / rec.mask_path)
        otsu_thr, states = _predict_family_states(rgb, otsu_polarity, km_centers, km_fg_clusters)
        ref_fg_px = int(np.sum(ref))
        n_pixels = int(ref.size)

        for family in DIAGNOSTIC_FAMILIES:
            st = states[family]
            m0, m1, opened, m2 = st["M0"], st["M1"], st["open"], st["M2"]

            met0 = binary_segmentation_metrics(m0, ref)
            met1 = binary_segmentation_metrics(m1, ref)
            met2 = binary_segmentation_metrics(m2, ref)

            areas0 = connected_component_areas(m0)
            areas1 = connected_component_areas(m1)
            areas2 = connected_component_areas(m2)
            area_chunks[(family, "M0")].append(areas0)
            area_chunks[(family, "M1")].append(areas1)
            area_chunks[(family, "M2")].append(areas2)

            rem_m1 = m0 & (~m1)
            rem_open = m1 & (~opened)
            add_close = m2 & (~opened)
            rem_m2_vs_m0 = m0 & (~m2)
            add_m2_vs_m0 = m2 & (~m0)

            rows.append(
                {
                    "sample_id": rec.sample_id,
                    "category": rec.category,
                    "group_id": rec.group_id,
                    "split": split,
                    "family": family,
                    "otsu_threshold": otsu_thr,
                    "ref_fg_px": ref_fg_px,
                    "ref_fg_pct": 100.0 * ref_fg_px / n_pixels,
                    "fg_px_M0": int(np.sum(m0)),
                    "fg_px_M1": int(np.sum(m1)),
                    "fg_px_M2": int(np.sum(m2)),
                    "fg_pct_M0": 100.0 * float(met0["predicted_foreground_fraction"]),
                    "fg_pct_M1": 100.0 * float(met1["predicted_foreground_fraction"]),
                    "fg_pct_M2": 100.0 * float(met2["predicted_foreground_fraction"]),
                    "cc_M0": int(len(areas0)),
                    "cc_M1": int(len(areas1)),
                    "cc_M2": int(len(areas2)),
                    "small_cc_M0": int(np.sum(areas0 < MIN_COMPONENT_AREA)),
                    "small_cc_M1": int(np.sum(areas1 < MIN_COMPONENT_AREA)),
                    "small_cc_M2": int(np.sum(areas2 < MIN_COMPONENT_AREA)),
                    "dice_M0": float(met0["dice"]),
                    "dice_M1": float(met1["dice"]),
                    "dice_M2": float(met2["dice"]),
                    "iou_M0": float(met0["iou"]),
                    "iou_M1": float(met1["iou"]),
                    "iou_M2": float(met2["iou"]),
                    "prec_M0": float(met0["precision"]),
                    "prec_M1": float(met1["precision"]),
                    "prec_M2": float(met2["precision"]),
                    "rec_M0": float(met0["recall"]),
                    "rec_M1": float(met1["recall"]),
                    "rec_M2": float(met2["recall"]),
                    "delta_dice_M1_M0": float(met1["dice"] - met0["dice"]),
                    "delta_dice_M2_M1": float(met2["dice"] - met1["dice"]),
                    "delta_dice_M2_M0": float(met2["dice"] - met0["dice"]),
                    "delta_iou_M1_M0": float(met1["iou"] - met0["iou"]),
                    "delta_iou_M2_M1": float(met2["iou"] - met1["iou"]),
                    "delta_iou_M2_M0": float(met2["iou"] - met0["iou"]),
                    "delta_prec_M1_M0": float(met1["precision"] - met0["precision"]),
                    "delta_prec_M2_M1": float(met2["precision"] - met1["precision"]),
                    "delta_rec_M1_M0": float(met1["recall"] - met0["recall"]),
                    "delta_rec_M2_M1": float(met2["recall"] - met1["recall"]),
                    "m1_rem_tp": int(np.sum(rem_m1 & ref)),
                    "m1_rem_fp": int(np.sum(rem_m1 & (~ref))),
                    "open_rem_tp": int(np.sum(rem_open & ref)),
                    "open_rem_fp": int(np.sum(rem_open & (~ref))),
                    "close_add_tp": int(np.sum(add_close & ref)),
                    "close_add_fp": int(np.sum(add_close & (~ref))),
                    "m02_rem_tp": int(np.sum(rem_m2_vs_m0 & ref)),
                    "m02_rem_fp": int(np.sum(rem_m2_vs_m0 & (~ref))),
                    "m02_add_tp": int(np.sum(add_m2_vs_m0 & ref)),
                    "m02_add_fp": int(np.sum(add_m2_vs_m0 & (~ref))),
                }
            )

    out_df = (
        pd.DataFrame(rows)
        .sort_values(["family", "sample_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    area_pools = {
        key: (np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32))
        for key, chunks in area_chunks.items()
    }

    if verify_against_saved and split == "val":
        _verify_val_cohort_parity(root, out_df)

    return out_df, area_pools


def _verify_val_cohort_parity(project_root: Path, val_df: pd.DataFrame) -> None:
    """Assert exact agreement against saved validation CSVs and morphology diagnostics."""
    saved_val = load_validation_results(project_root)
    saved_diag = load_development_morphology_csv(project_root)
    saved_diag_val = saved_diag.loc[saved_diag["split"].eq("val")]

    for family in DIAGNOSTIC_FAMILIES:
        live_sub = val_df.loc[val_df["family"].eq(family)].set_index("sample_id").sort_index()
        for stage in DIAGNOSTIC_STAGES:
            method_id = f"{family}_{stage}"
            ref_scores = (
                saved_val.loc[saved_val["method"].eq(method_id)]
                .set_index("sample_id")
                .reindex(live_sub.index)
            )
            np.testing.assert_allclose(
                live_sub[f"dice_{stage}"].to_numpy(dtype=np.float64),
                ref_scores["dice"].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                live_sub[f"iou_{stage}"].to_numpy(dtype=np.float64),
                ref_scores["iou"].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-10,
            )

            ref_diag = (
                saved_diag_val.loc[saved_diag_val["method"].eq(method_id)]
                .set_index("sample_id")
                .reindex(live_sub.index)
            )
            np.testing.assert_array_equal(
                live_sub[f"cc_{stage}"].to_numpy(dtype=np.int64),
                ref_diag["component_count"].to_numpy(dtype=np.int64),
            )
            np.testing.assert_array_equal(
                live_sub[f"fg_px_{stage}"].to_numpy(dtype=np.int64),
                ref_diag["foreground_pixels"].to_numpy(dtype=np.int64),
            )


def classify_delta_outcomes(
    deltas: pd.Series,
    negligible_threshold: float = NEGLIGIBLE_DELTA_THRESHOLD,
) -> pd.Series:
    """Bucket paired metric changes into improvement, negligible, or regression."""
    arr = deltas.to_numpy(dtype=np.float64)
    labels = np.where(
        arr >= negligible_threshold,
        OUTCOME_LABELS[0],
        np.where(arr <= -negligible_threshold, OUTCOME_LABELS[2], OUTCOME_LABELS[1]),
    )
    return pd.Series(labels, index=deltas.index, dtype="object")


def select_deterministic_representative_cases(val_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministically select representative improvements, regressions, and negligible cases."""
    specs = [
        ("Otsu: Max Gain (M2−M0)", "otsu", "delta_dice_M2_M0", "max"),
        ("Otsu: Max Polyp Regress (M2−M1)", "otsu", "delta_dice_M2_M1", "min_polyp"),
        ("Otsu: Max Regress (M2−M0)", "otsu", "delta_dice_M2_M0", "min"),
        ("K-Means k=4: Max Gain (M2−M0)", "k4", "delta_dice_M2_M0", "max"),
        ("K-Means k=4: Max Regress (M2−M0)", "k4", "delta_dice_M2_M0", "min"),
        ("Otsu: Negligible |ΔDice| (M2−M0)", "otsu", "delta_dice_M2_M0", "min_abs_nonzero"),
        ("K-Means k=4: Negligible |ΔDice| (M1−M0)", "k4", "delta_dice_M1_M0", "min_abs_nonzero"),
    ]
    rows: list[dict[str, object]] = []
    for role, family, col, rule in specs:
        sub = val_df.loc[val_df["family"].eq(family)].copy()
        if rule == "min_polyp":
            sub = sub.loc[sub["category"].eq("Polyp")].copy()
        if rule == "max":
            picked = sub.sort_values([col, "sample_id"], ascending=[False, True]).iloc[0]
        elif rule in ("min", "min_polyp"):
            picked = sub.sort_values([col, "sample_id"], ascending=[True, True]).iloc[0]
        elif rule == "min_abs_nonzero":
            sub = sub.loc[sub[col].abs() > 1e-12].copy()
            sub["_abs"] = sub[col].abs()
            picked = sub.sort_values(["_abs", "sample_id"], ascending=[True, True]).iloc[0]
        else:
            raise ValueError(f"Unknown selection rule: {rule}")

        rows.append(
            {
                "role": role,
                "family": family,
                "sample_id": picked["sample_id"],
                "category": picked["category"],
                "dice_M0": float(picked["dice_M0"]),
                "dice_M1": float(picked["dice_M1"]),
                "dice_M2": float(picked["dice_M2"]),
                "delta_M1_M0": float(picked["delta_dice_M1_M0"]),
                "delta_M2_M1": float(picked["delta_dice_M2_M1"]),
                "delta_M2_M0": float(picked["delta_dice_M2_M0"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_transitions_by_category(val_df: pd.DataFrame) -> pd.DataFrame:
    """Compute category-wise and macro-averaged Dice, IoU, Precision, Recall, and outcomes."""
    rows: list[dict[str, object]] = []
    for family in DIAGNOSTIC_FAMILIES:
        sub = val_df.loc[val_df["family"].eq(family)]
        cat_rows: list[dict[str, object]] = []
        for cat in EXPECTED_CATEGORIES:
            c = sub.loc[sub["category"].eq(cat)]
            outcomes = classify_delta_outcomes(c["delta_dice_M2_M0"])
            rec = {
                "family": family,
                "category": cat,
                "n": int(len(c)),
                "dice_M0": float(c["dice_M0"].mean()),
                "dice_M1": float(c["dice_M1"].mean()),
                "dice_M2": float(c["dice_M2"].mean()),
                "d_dice_M1_M0": float(c["delta_dice_M1_M0"].mean()),
                "d_dice_M2_M1": float(c["delta_dice_M2_M1"].mean()),
                "d_dice_M2_M0": float(c["delta_dice_M2_M0"].mean()),
                "d_iou_M2_M0": float(c["delta_iou_M2_M0"].mean()),
                "d_prec_M1_M0": float(c["delta_prec_M1_M0"].mean()),
                "d_prec_M2_M1": float(c["delta_prec_M2_M1"].mean()),
                "d_rec_M1_M0": float(c["delta_rec_M1_M0"].mean()),
                "d_rec_M2_M1": float(c["delta_rec_M2_M1"].mean()),
                "fg_pct_M0": float(c["fg_pct_M0"].mean()),
                "fg_pct_M1": float(c["fg_pct_M1"].mean()),
                "fg_pct_M2": float(c["fg_pct_M2"].mean()),
                "cc_M0": float(c["cc_M0"].mean()),
                "cc_M1": float(c["cc_M1"].mean()),
                "cc_M2": float(c["cc_M2"].mean()),
                "improve_n": int(outcomes.eq(OUTCOME_LABELS[0]).sum()),
                "negligible_n": int(outcomes.eq(OUTCOME_LABELS[1]).sum()),
                "regress_n": int(outcomes.eq(OUTCOME_LABELS[2]).sum()),
            }
            cat_rows.append(rec)
            rows.append(rec)

        cat_df = pd.DataFrame(cat_rows)
        macro = {"family": family, "category": "Macro (6 cat)", "n": int(len(sub))}
        for col in (
            "dice_M0",
            "dice_M1",
            "dice_M2",
            "d_dice_M1_M0",
            "d_dice_M2_M1",
            "d_dice_M2_M0",
            "d_iou_M2_M0",
            "d_prec_M1_M0",
            "d_prec_M2_M1",
            "d_rec_M1_M0",
            "d_rec_M2_M1",
            "fg_pct_M0",
            "fg_pct_M1",
            "fg_pct_M2",
            "cc_M0",
            "cc_M1",
            "cc_M2",
        ):
            macro[col] = float(cat_df[col].mean())
        for col in ("improve_n", "negligible_n", "regress_n"):
            macro[col] = int(cat_df[col].sum())
        rows.append(macro)
    return pd.DataFrame(rows)


def _load_case_bundle(
    project_root: str | Path,
    sample_id: str,
) -> tuple[pd.Series, np.ndarray, np.ndarray, float, dict[str, dict[str, np.ndarray]]]:
    """Load manifest record, RGB, GT mask, and M0/M1/OPEN/M2 masks for one dev image."""
    root = Path(project_root).expanduser().resolve()
    manifest = load_split_manifest(root, verify=True)
    matches = manifest.loc[manifest["sample_id"].eq(sample_id)]
    if len(matches) != 1:
        raise KeyError(f"Expected unique manifest record for '{sample_id}', found {len(matches)}.")
    rec = matches.iloc[0]
    if rec["split"] == "test":
        raise ValueError(f"Forbidden test-split sample requested: '{sample_id}'.")

    otsu_cfg = load_otsu_config(root)
    km_model = load_kmeans_model(root, k=4, verify_provenance=True)
    rgb = load_rgb(root / str(rec["image_path"]))
    ref = load_reference_mask(root / str(rec["mask_path"]))
    otsu_thr, states = _predict_family_states(
        rgb,
        str(otsu_cfg["foreground_polarity"]),
        np.asarray(km_model["cluster_centers"], dtype=np.float64),
        np.asarray(km_model["foreground_clusters"], dtype=np.int64),
    )
    return rec, rgb, ref, otsu_thr, states


def plot_seed_baseline_grid(
    project_root: str | Path,
    sample_ids: Iterable[str],
) -> Figure:
    """Plot RGB, GT reference mask, and baseline M0 masks for the seed diagnostic cases."""
    configure_notebook_style()
    ids = list(sample_ids)
    fig, axes = plt.subplots(
        len(ids),
        4,
        figsize=(11.4, 2.85 * len(ids)),
        layout="constrained",
    )
    if len(ids) == 1:
        axes = axes[np.newaxis, :]

    col_headers = ("Original RGB", "GT Reference", "Otsu M0 (Raw)", "K-Means k=4 M0 (Raw)")
    for row_idx, sample_id in enumerate(ids):
        rec, rgb, ref, otsu_thr, states = _load_case_bundle(project_root, sample_id)
        ref_u8 = ref.astype(np.uint8)
        o_m0 = states["otsu"]["M0"]
        k_m0 = states["k4"]["M0"]
        o_met = binary_segmentation_metrics(o_m0, ref)
        k_met = binary_segmentation_metrics(k_m0, ref)

        panels = [
            (rgb, None, f"thr={otsu_thr:.0f}"),
            (ref_u8, "gray", f"FG {100.0 * float(ref.mean()):.1f}%"),
            (
                o_m0.astype(np.uint8),
                "gray",
                f"Dice {o_met['dice']:.3f} | FG {100.0 * float(o_m0.mean()):.1f}%",
            ),
            (
                k_m0.astype(np.uint8),
                "gray",
                f"Dice {k_met['dice']:.3f} | FG {100.0 * float(k_m0.mean()):.1f}%",
            ),
        ]
        for col_idx, (img, cmap, xlbl) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if cmap is None:
                ax.imshow(img, interpolation="nearest")
            else:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
            _style_image_axis(ax)
            ax.set_xlabel(xlbl, fontsize=8.4)
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=9.8, fontweight="bold")
        axes[row_idx, 0].set_ylabel(
            f"{sample_id.split('/', 1)[1]}\n({rec['category']})",
            fontsize=8.6,
            fontweight="bold",
        )
    return fig


def plot_case_morphology_progression(
    project_root: str | Path,
    sample_id: str,
) -> Figure:
    """Compare Otsu & K-Means across M0/M1/M2 masks, pixel transitions, and TP/FP/FN overlays."""
    configure_notebook_style()
    rec, rgb, ref, _, states = _load_case_bundle(project_root, sample_id)

    fig, axes = plt.subplots(2, 6, figsize=(15.6, 5.5), layout="constrained")
    col_headers = (
        "M0 (<16 px coral)",
        "M1 (≥16 px)",
        "M2 (<16 px coral)",
        "Perubahan M0→M2",
        "M0 Overlay",
        "M2 Overlay",
    )

    for row_idx, family in enumerate(DIAGNOSTIC_FAMILIES):
        st = states[family]
        m0, m1, m2 = st["M0"], st["M1"], st["M2"]

        m0_img, cc0, small0 = render_component_highlight_rgb(m0)
        m1_img, cc1, _ = render_component_highlight_rgb(m1)
        m2_img, cc2, small2 = render_component_highlight_rgb(m2)
        diff_img, diff_px = render_transition_diff_rgb(m0, m1, m2)
        ov0, _ = build_error_overlay(rgb, m0, ref)
        ov2, _ = build_error_overlay(rgb, m2, ref)

        met0 = binary_segmentation_metrics(m0, ref)
        met1 = binary_segmentation_metrics(m1, ref)
        met2 = binary_segmentation_metrics(m2, ref)
        delta_20 = met2["dice"] - met0["dice"]

        panels = [
            (m0_img, f"FG {100.0 * m0.mean():.1f}% | C {cc0} ({small0}<16)"),
            (m1_img, f"FG {100.0 * m1.mean():.1f}% | C {cc1} | D {met1['dice']:.3f}"),
            (m2_img, f"FG {100.0 * m2.mean():.1f}% | C {cc2} ({small2}<16)"),
            (
                diff_img,
                f"−{diff_px['removed_m1_px']} M1 | −{diff_px['removed_m2_px']} / +{diff_px['added_m2_px']} M2",
            ),
            (ov0, f"D {met0['dice']:.3f} | P {met0['precision']:.2f} R {met0['recall']:.2f}"),
            (ov2, f"D {met2['dice']:.3f} ({delta_20:+.3f}) | P {met2['precision']:.2f} R {met2['recall']:.2f}"),
        ]
        for col_idx, (img, xlbl) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, interpolation="nearest")
            _style_image_axis(ax)
            ax.set_xlabel(xlbl, fontsize=7.9)
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=9.3, fontweight="bold")

        axes[row_idx, 0].set_ylabel(
            "Otsu" if family == "otsu" else "K-Means k=4",
            fontsize=9.5,
            fontweight="bold",
        )

    fig.suptitle(
        f"Efek Morfologi M0 → M1 → M2: {sample_id} ({rec['split']})",
        fontsize=11.0,
        fontweight="bold",
    )
    legend_handles = [
        Patch(facecolor=MASK_KEPT_RGB / 255.0, label="FG tetap / ≥16 px"),
        Patch(facecolor=MASK_SMALL_RGB / 255.0, label="<16 px / dihapus M1"),
        Patch(facecolor=DIFF_REMOVED_M2_RGB / 255.0, label="Dihapus M2 (OPEN)"),
        Patch(facecolor=DIFF_ADDED_M2_RGB / 255.0, label="Ditambah M2 (CLOSE)"),
        Patch(facecolor=TP_COLOR_RGB / 255.0, label="TP"),
        Patch(facecolor=FP_COLOR_RGB / 255.0, label="FP"),
        Patch(facecolor=FN_COLOR_RGB / 255.0, label="FN"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=7,
        frameon=True,
        fontsize=8.2,
    )
    return fig


def plot_representative_cases_grid(
    project_root: str | Path,
    case_specs: list[tuple[str, str, str]],
) -> Figure:
    """Plot a compact grid of deterministic improvements and regressions [(role, family, sample_id)]."""
    configure_notebook_style()
    n_rows = len(case_specs)
    fig, axes = plt.subplots(n_rows, 5, figsize=(13.6, 2.62 * n_rows), layout="constrained")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = (
        "RGB + Kontur GT",
        "M0 (<16 px coral)",
        "M1 (≥16 px)",
        "M2 (<16 px coral)",
        "Perubahan M0→M2",
    )
    for row_idx, (role, family, sample_id) in enumerate(case_specs):
        _, rgb, ref, _, states = _load_case_bundle(project_root, sample_id)
        st = states[family]
        m0, m1, m2 = st["M0"], st["M1"], st["M2"]

        m0_img, cc0, small0 = render_component_highlight_rgb(m0)
        m1_img, cc1, _ = render_component_highlight_rgb(m1)
        m2_img, cc2, small2 = render_component_highlight_rgb(m2)
        diff_img, diff_px = render_transition_diff_rgb(m0, m1, m2)

        d0 = binary_segmentation_metrics(m0, ref)["dice"]
        d1 = binary_segmentation_metrics(m1, ref)["dice"]
        d2 = binary_segmentation_metrics(m2, ref)["dice"]

        panels = [
            (render_rgb_with_gt_contour(rgb, ref), f"{sample_id.split('/', 1)[1]}"),
            (m0_img, f"Dice {d0:.3f} | C {cc0} ({small0}<16)"),
            (m1_img, f"Dice {d1:.3f} ({d1 - d0:+.3f}) | C {cc1}"),
            (m2_img, f"Dice {d2:.3f} ({d2 - d1:+.3f}) | C {cc2} ({small2}<16)"),
            (diff_img, f"−{diff_px['removed_m1_px']} M1 | −{diff_px['removed_m2_px']}/+{diff_px['added_m2_px']}"),
        ]
        for col_idx, (img, xlbl) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, interpolation="nearest")
            _style_image_axis(ax)
            ax.set_xlabel(xlbl, fontsize=7.9)
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=9.2, fontweight="bold")
        axes[row_idx, 0].set_ylabel(role, fontsize=8.3, fontweight="bold")

    legend_handles = [
        Patch(facecolor=GT_CONTOUR_RGB / 255.0, label="Kontur GT"),
        Patch(facecolor=MASK_KEPT_RGB / 255.0, label="FG ≥16 px / tetap"),
        Patch(facecolor=MASK_SMALL_RGB / 255.0, label="<16 px / dihapus M1"),
        Patch(facecolor=DIFF_REMOVED_M2_RGB / 255.0, label="Dihapus M2 (OPEN)"),
        Patch(facecolor=DIFF_ADDED_M2_RGB / 255.0, label="Ditambah M2 (CLOSE)"),
    ]
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=5, frameon=True, fontsize=8.3)
    return fig


def _bin_area_percentages(areas: np.ndarray) -> np.ndarray:
    """Compute share (%) of connected components across fixed logarithmic area buckets."""
    if len(areas) == 0:
        return np.zeros(len(AREA_BIN_LABELS), dtype=np.float64)
    counts, _ = np.histogram(areas, bins=np.asarray(AREA_BIN_EDGES, dtype=np.int64))
    return 100.0 * counts.astype(np.float64) / float(len(areas))


def plot_component_distributions(
    area_pools: dict[tuple[str, str], np.ndarray],
    val_df: pd.DataFrame,
    dev_diag_df: pd.DataFrame,
) -> Figure:
    """Plot component area distributions (<16 px cutoff) and component counts across dev splits."""
    configure_notebook_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4), layout="constrained")
    x = np.arange(len(AREA_BIN_LABELS), dtype=np.float64)
    width = 0.25

    for panel_idx, family in enumerate(DIAGNOSTIC_FAMILIES):
        ax = axes[panel_idx]
        for s_idx, stage in enumerate(DIAGNOSTIC_STAGES):
            pcts = _bin_area_percentages(area_pools[(family, stage)])
            ax.bar(
                x + (s_idx - 1) * width,
                pcts,
                width=width,
                color=STAGE_COLORS[stage],
                label=stage,
                alpha=0.9,
            )
        ax.axvline(2.5, color="#C0392B", linestyle="--", linewidth=1.4, label="Batas M1 = 16 px")
        ax.set_xticks(x)
        ax.set_xticklabels(AREA_BIN_LABELS, fontsize=8.4)
        ax.set_xlabel("Luas Komponen Terhubung (piksel)", fontsize=9.2)
        ax.set_ylabel("Proporsi Komponen (%)" if panel_idx == 0 else "", fontsize=9.2)
        title_fam = "Otsu" if family == "otsu" else "K-Means k=4"
        ax.set_title(f"Distribusi Luas Komponen — {title_fam} (val)", fontsize=9.8, fontweight="bold")
        ax.set_ylim(0, 56)
        ax.grid(axis="y", alpha=0.28)
        ax.legend(fontsize=8.0, frameon=True)

    # Panel 2: Mean components per image across train (saved) vs val (live/saved)
    ax_c = axes[2]
    dev_means = (
        dev_diag_df.groupby(["split", "family", "stage"], observed=True)["component_count"]
        .mean()
        .unstack("split")
    )
    methods_order = [(fam, st) for fam in DIAGNOSTIC_FAMILIES for st in DIAGNOSTIC_STAGES]
    mx = np.arange(len(methods_order), dtype=np.float64)
    bar_w = 0.35
    train_vals = [float(dev_means.loc[(fam, st), "train"]) for fam, st in methods_order]
    val_vals = [float(dev_means.loc[(fam, st), "val"]) for fam, st in methods_order]

    ax_c.bar(mx - bar_w / 2, train_vals, width=bar_w, color="#5D6D7E", label="train (n=1.535)")
    ax_c.bar(mx + bar_w / 2, val_vals, width=bar_w, color="#1F77B4", label="val (n=372)")
    for idx, val_v in enumerate(val_vals):
        ax_c.text(idx + bar_w / 2, val_v + 3.0, f"{val_v:.1f}", ha="center", va="bottom", fontsize=7.8)
    ax_c.set_ylim(0, max(max(train_vals), max(val_vals)) * 1.15)
    ax_c.set_xticks(mx)
    ax_c.set_xticklabels([method_label(f"{f}_{s}").replace(" ", "\n", 1) for f, s in methods_order], fontsize=7.8)
    ax_c.set_ylabel("Rata-rata Komponen / Citra", fontsize=9.2)
    ax_c.set_title("Perubahan Jumlah Komponen (train vs val)", fontsize=9.8, fontweight="bold")
    ax_c.grid(axis="y", alpha=0.28)
    ax_c.legend(fontsize=8.2, frameon=True)
    return fig


def plot_coverage_and_pixel_flips(val_df: pd.DataFrame) -> Figure:
    """Plot foreground coverage by category and per-step TP/FP pixel removals and additions."""
    configure_notebook_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.7), layout="constrained")

    # Left: Category foreground coverage (%) vs GT reference
    ax0 = axes[0]
    cats = list(EXPECTED_CATEGORIES)
    cx = np.arange(len(cats), dtype=np.float64)
    bw = 0.16
    otsu_sub = val_df.loc[val_df["family"].eq("otsu")].groupby("category", observed=True).mean(numeric_only=True)
    k4_sub = val_df.loc[val_df["family"].eq("k4")].groupby("category", observed=True).mean(numeric_only=True)

    series = [
        ("GT Ref", [float(otsu_sub.loc[c, "ref_fg_pct"]) for c in cats], "#2C3E50"),
        ("Otsu M0", [float(otsu_sub.loc[c, "fg_pct_M0"]) for c in cats], "#5DADE2"),
        ("Otsu M2", [float(otsu_sub.loc[c, "fg_pct_M2"]) for c in cats], "#1F77B4"),
        ("k4 M0", [float(k4_sub.loc[c, "fg_pct_M0"]) for c in cats], "#F5B041"),
        ("k4 M2", [float(k4_sub.loc[c, "fg_pct_M2"]) for c in cats], "#D35400"),
    ]
    for idx, (lbl, vals, color) in enumerate(series):
        ax0.bar(cx + (idx - 2.0) * bw, vals, width=bw, label=lbl, color=color)
    ax0.set_xticks(cx)
    ax0.set_xticklabels(cats, rotation=18, ha="right", fontsize=8.4)
    ax0.set_ylabel("Cakupan Foreground (%)", fontsize=9.3)
    ax0.set_ylim(0, 102)
    ax0.set_title("Cakupan Foreground per Kategori (val)", fontsize=9.9, fontweight="bold")
    ax0.grid(axis="y", alpha=0.28)
    ax0.legend(ncol=3, fontsize=7.9, frameon=True)

    # Right: Step-wise pixel flips per image decomposed into TP vs FP pixels
    ax1 = axes[1]
    step_labels = [
        "Otsu\nM0→M1\n(−)",
        "Otsu\nOPEN\n(−)",
        "Otsu\nCLOSE\n(+)",
        "k4\nM0→M1\n(−)",
        "k4\nOPEN\n(−)",
        "k4\nCLOSE\n(+)",
    ]
    sx = np.arange(len(step_labels), dtype=np.float64)
    o_m = val_df.loc[val_df["family"].eq("otsu")].mean(numeric_only=True)
    k_m = val_df.loc[val_df["family"].eq("k4")].mean(numeric_only=True)

    # Signed values: removals negative, additions positive
    tp_vals = np.array([
        -o_m["m1_rem_tp"],
        -o_m["open_rem_tp"],
        o_m["close_add_tp"],
        -k_m["m1_rem_tp"],
        -k_m["open_rem_tp"],
        k_m["close_add_tp"],
    ])
    fp_vals = np.array([
        -o_m["m1_rem_fp"],
        -o_m["open_rem_fp"],
        o_m["close_add_fp"],
        -k_m["m1_rem_fp"],
        -k_m["open_rem_fp"],
        k_m["close_add_fp"],
    ])
    ax1.bar(sx, tp_vals, width=0.55, color="#8E44AD", alpha=0.88, label="Piksel TP (− hilang, + pulih)")
    ax1.bar(sx, fp_vals, width=0.55, bottom=tp_vals, color="#E67E22", alpha=0.88, label="Piksel FP (− bersih, + meluas)")
    ax1.axhline(0.0, color="#2C3E50", linewidth=1.0)
    for idx, (tv, fv) in enumerate(zip(tp_vals, fp_vals)):
        tot = tv + fv
        y_pos = tot + (35.0 if tot >= 0 else -45.0)
        va = "bottom" if tot >= 0 else "top"
        ax1.text(idx, y_pos, f"{int(round(abs(tot)))}\n(TP {int(round(abs(tv)))})", ha="center", va=va, fontsize=7.6)
    ax1.set_xticks(sx)
    ax1.set_xticklabels(step_labels, fontsize=8.3)
    ax1.set_ylabel("Rata-rata Piksel Berubah / Citra", fontsize=9.3)
    ax1.set_ylim(-2300, 1950)
    ax1.set_title("Perubahan Piksel per Tahap: Komposisi TP vs FP (val)", fontsize=9.9, fontweight="bold")
    ax1.grid(axis="y", alpha=0.28)
    ax1.legend(loc="lower right", fontsize=8.0, frameon=True)
    return fig


def plot_metric_distributions_and_tradeoffs(val_df: pd.DataFrame) -> Figure:
    """Plot paired Dice deltas, category outcome proportions, category deltas, and Precision-Recall shift."""
    configure_notebook_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.2), layout="constrained")

    # 1. Top-left: Paired Delta Dice boxplot across stages and families
    ax0 = axes[0, 0]
    box_pairs = [
        ("Otsu\nM1−M0", "otsu", "delta_dice_M1_M0"),
        ("Otsu\nM2−M1", "otsu", "delta_dice_M2_M1"),
        ("Otsu\nM2−M0", "otsu", "delta_dice_M2_M0"),
        ("k4\nM1−M0", "k4", "delta_dice_M1_M0"),
        ("k4\nM2−M1", "k4", "delta_dice_M2_M1"),
        ("k4\nM2−M0", "k4", "delta_dice_M2_M0"),
    ]
    data = [val_df.loc[val_df["family"].eq(fam), col].to_numpy() for _, fam, col in box_pairs]
    ax0.axhspan(-NEGLIGIBLE_DELTA_THRESHOLD, NEGLIGIBLE_DELTA_THRESHOLD, color="#EAEDED", alpha=0.85, label="Zona negligible (|Δ| < 0.005)")
    ax0.axhline(0.0, color="#566573", linestyle="--", linewidth=1.0)
    bp = ax0.boxplot(data, patch_artist=True, showfliers=True, flierprops={"markersize": 3.0, "alpha": 0.45})
    for idx, box in enumerate(bp["boxes"]):
        fam = box_pairs[idx][1]
        box.set_facecolor(FAMILY_COLORS[fam])
        box.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.5)
    ax0.set_xticks(np.arange(1, len(box_pairs) + 1))
    ax0.set_xticklabels([lbl for lbl, _, _ in box_pairs], fontsize=8.4)
    ax0.set_ylabel("ΔDice Berpasangan", fontsize=9.3)
    ax0.set_title("Distribusi ΔDice Per-Citra (val, n=372)", fontsize=9.9, fontweight="bold")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(loc="lower left", fontsize=8.0, frameon=True)

    # 2. Top-right: Category outcome proportions for M2 - M0 (Otsu vs k4)
    ax1 = axes[0, 1]
    cats = list(EXPECTED_CATEGORIES)
    y_rows: list[tuple[str, pd.Series]] = []
    for cat in cats:
        for fam, fname in (("otsu", "Otsu"), ("k4", "k4")):
            sub_d = val_df.loc[val_df["family"].eq(fam) & val_df["category"].eq(cat), "delta_dice_M2_M0"]
            y_rows.append((f"{cat} [{fname}]", classify_delta_outcomes(sub_d)))
    yy = np.arange(len(y_rows), dtype=np.float64)
    left = np.zeros(len(y_rows), dtype=np.float64)
    for outcome in OUTCOME_LABELS:
        shares = np.array([100.0 * float(s.eq(outcome).mean()) for _, s in y_rows])
        ax1.barh(yy, shares, left=left, height=0.72, color=OUTCOME_COLORS[outcome], label=outcome)
        left += shares
    ax1.set_yticks(yy)
    ax1.set_yticklabels([lbl for lbl, _ in y_rows], fontsize=8.0)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Proporsi Citra (%)", fontsize=9.3)
    ax1.set_title("Klasifikasi Dampak Akhir (M2−M0) per Kategori", fontsize=9.9, fontweight="bold", pad=20)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.00), ncol=3, fontsize=7.6, frameon=False)

    # 3. Bottom-left: Category Mean Delta Dice M1-M0 vs M2-M1
    ax2 = axes[1, 0]
    cx = np.arange(len(cats), dtype=np.float64)
    bw = 0.2
    o_cat = val_df.loc[val_df["family"].eq("otsu")].groupby("category", observed=True).mean(numeric_only=True)
    k_cat = val_df.loc[val_df["family"].eq("k4")].groupby("category", observed=True).mean(numeric_only=True)
    bars = [
        ("Otsu M1−M0", [float(o_cat.loc[c, "delta_dice_M1_M0"]) for c in cats], "#7FB3D5"),
        ("Otsu M2−M1", [float(o_cat.loc[c, "delta_dice_M2_M1"]) for c in cats], "#1F77B4"),
        ("k4 M1−M0", [float(k_cat.loc[c, "delta_dice_M1_M0"]) for c in cats], "#F5B041"),
        ("k4 M2−M1", [float(k_cat.loc[c, "delta_dice_M2_M1"]) for c in cats], "#D35400"),
    ]
    for idx, (lbl, vals, color) in enumerate(bars):
        ax2.bar(cx + (idx - 1.5) * bw, vals, width=bw, color=color, label=lbl)
    ax2.axhline(0.0, color="#2C3E50", linewidth=1.0)
    ax2.set_xticks(cx)
    ax2.set_xticklabels(cats, rotation=18, ha="right", fontsize=8.4)
    ax2.set_ylabel("Rata-rata ΔDice", fontsize=9.3)
    ax2.set_title("Pergeseran Rata-rata ΔDice per Kategori (M1−M0 vs M2−M1)", fontsize=9.9, fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(ncol=2, fontsize=8.0, frameon=True)

    # 4. Bottom-right: Delta Precision vs Delta Recall quadrant per category (M2-M1 & Otsu M1-M0)
    ax3 = axes[1, 1]
    ax3.axhline(0.0, color="#7F8C8D", linestyle="--", linewidth=0.9)
    ax3.axvline(0.0, color="#7F8C8D", linestyle="--", linewidth=0.9)
    for fam, marker, label_prefix in (("otsu", "o", "Otsu M2−M1"), ("k4", "^", "k4 M2−M1")):
        sub = val_df.loc[val_df["family"].eq(fam)].groupby("category", observed=True).mean(numeric_only=True)
        for cat in cats:
            dx = float(sub.loc[cat, "delta_prec_M2_M1"])
            dy = float(sub.loc[cat, "delta_rec_M2_M1"])
            color = FAMILY_COLORS[fam]
            ax3.scatter(dx, dy, s=64, marker=marker, color=color, edgecolor="#1C2833", linewidth=0.7, zorder=4)
            short_cat = cat.replace("Adenocarcinoma", "Adeno").replace("High-grade IN", "HGIN").replace("Low-grade IN", "LGIN").replace("Serrated adenoma", "Serr")
            offset_xy = (5, -9) if (fam == "k4" and cat == "Polyp") else ((-4, 7) if (fam == "k4" and cat == "Normal") else (4, 3))
            ax3.annotate(
                short_cat,
                (dx, dy),
                xytext=offset_xy,
                textcoords="offset points",
                fontsize=7.6,
                color=color,
            )
    # Show Otsu M1-M0 mean point for reference
    o_m1 = val_df.loc[val_df["family"].eq("otsu")]
    ax3.scatter(
        float(o_m1["delta_prec_M1_M0"].mean()),
        float(o_m1["delta_rec_M1_M0"].mean()),
        s=90,
        marker="s",
        color="#2EA043",
        edgecolor="#1C2833",
        zorder=5,
        label="Rata-rata Otsu M1−M0",
    )
    ax3.scatter([], [], s=64, marker="o", color=FAMILY_COLORS["otsu"], edgecolor="#1C2833", label="Otsu M2−M1 (per kategori)")
    ax3.scatter([], [], s=64, marker="^", color=FAMILY_COLORS["k4"], edgecolor="#1C2833", label="k4 M2−M1 (per kategori)")
    ax3.set_xlim(-0.0108, 0.0180)
    ax3.set_xlabel("ΔPrecision", fontsize=9.3)
    ax3.set_ylabel("ΔRecall", fontsize=9.3)
    ax3.set_title("Trade-off ΔPrecision vs ΔRecall per Kategori", fontsize=9.9, fontweight="bold")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="lower left", fontsize=7.9, frameon=True)
    return fig
