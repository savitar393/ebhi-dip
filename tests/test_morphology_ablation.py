"""Tests for Task 2 controlled morphology ablation operations, strict CV, and leak-free parity."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from ebhi_dip.classification import CV_FOLDS
from ebhi_dip.dataset import EXPECTED_CATEGORIES, load_split_manifest
from ebhi_dip.features import APPEARANCE_FEATURE_NAMES, MORPHOLOGY_FEATURE_NAMES
from ebhi_dip.morphology_ablation import (
    ALL_VARIANTS,
    BASELINE_VARIANTS,
    CANDIDATE_VARIANTS,
    CROSS_3X3_KERNEL,
    MAX_SMALL_HOLE_AREA,
    MIN_COMPONENT_AREA,
    PREDECLARED_ANCHOR_SPECS,
    ablation_morphology_variants,
    build_strict_train_folds,
    compute_morph_feature_shifts,
    enclosed_hole_stats,
    evaluate_anchor_cv_grid,
    fill_small_enclosed_holes,
    filter_small_components,
    load_saved_ablation_outputs,
    plot_ablation_cv_macro_f1,
    plot_ablation_delta_dice_distributions,
    plot_ablation_feature_shifts,
    plot_ablation_per_category_f1,
    plot_ablation_pixel_flips_and_components,
    plot_ablation_segmentation_overview,
    summarize_ablation_segmentation,
)
from ebhi_dip.paths import find_project_root
from ebhi_dip.plotting import finish_figure
from ebhi_dip.segmentation import morphology_variants
from ebhi_dip.strict_cv import CLASSIFICATION_2X3_STRICT_DIR_REL

REPO_ROOT = find_project_root()


def test_filter_small_components_shape_dtype_empty_full_and_edges() -> None:
    empty = np.zeros((17, 31), dtype=np.uint8)
    out_empty = filter_small_components(empty)
    assert out_empty.shape == (17, 31)
    assert out_empty.dtype == bool
    assert not out_empty.any()

    # Tiny 3x3 all-True mask (< 16 px) must cleanly empty after area filter.
    tiny_full = np.ones((3, 3), dtype=bool)
    out_tiny = filter_small_components(tiny_full, min_area=16)
    assert out_tiny.shape == (3, 3)
    assert out_tiny.dtype == bool
    assert not out_tiny.any()

    # Full 10x10 mask (100 px >= 16) stays completely True.
    full = np.ones((10, 10), dtype=bool)
    out_full = filter_small_components(full, min_area=16)
    assert np.array_equal(out_full, full)

    # Connectivity boundary: 15-px island dropped, 16-px island kept, diagonal 16-px chain kept under 8-conn.
    mask = np.zeros((30, 30), dtype=bool)
    mask[1:4, 1:6] = True  # 3x5 = 15 px -> removed
    mask[6:10, 1:5] = True  # 4x4 = 16 px -> kept
    for i in range(16):
        mask[12 + i, 12 + i] = True  # 16 diagonally connected px -> kept under 8-conn, dropped under 4-conn

    kept8 = filter_small_components(mask, min_area=16, connectivity=8)
    assert not kept8[1:4, 1:6].any()
    assert kept8[6:10, 1:5].all()
    assert all(bool(kept8[12 + i, 12 + i]) for i in range(16))

    kept4 = filter_small_components(mask, min_area=16, connectivity=4)
    assert kept4[6:10, 1:5].all()
    assert not any(bool(kept4[12 + i, 12 + i]) for i in range(16))


def test_fill_small_enclosed_holes_edge_cases_and_connectivity() -> None:
    empty = np.zeros((16, 16), dtype=bool)
    full = np.ones((16, 16), dtype=bool)
    assert np.array_equal(fill_small_enclosed_holes(empty), empty)
    assert np.array_equal(fill_small_enclosed_holes(full), full)

    # Diagonal 8-connected ring enclosing a single interior background hole at (5, 5).
    ring = np.zeros((16, 20), dtype=bool)
    ring[4:7, 4:7] = True
    ring[4, 4] = ring[4, 6] = ring[6, 4] = ring[6, 6] = False
    ring[5, 5] = False  # 1-px hole enclosed only under 4-connected background!
    # Interior 15-px hole (must be filled at threshold 15)
    ring[2:13, 8:15] = True
    ring[4:7, 9:14] = False  # 3x5 = 15 px enclosed hole
    # Interior 16-px hole (must be preserved at threshold 15)
    ring[1:14, 15:20] = True
    ring[3:7, 16:20] = False  # touches col=19 border! Wait, keep interior col=15..19 -> shift col 19 off border:
    ring = np.zeros((20, 30), dtype=bool)
    # 1. Diagonal 8-ring around (4, 4)
    for r, c in ((3, 4), (4, 3), (4, 5), (5, 4)):
        ring[r, c] = True
    # 2. Exact 15-px enclosed hole
    ring[2:9, 8:15] = True
    ring[3:6, 9:14] = False  # 3x5 = 15 px
    # 3. Exact 16-px enclosed hole
    ring[2:10, 17:24] = True
    ring[3:7, 18:22] = False  # 4x4 = 16 px
    # 4. Border-touching 3-px bay at top edge (row=0) -> NEVER fill!
    ring[0:3, 25:28] = True
    ring[0, 26] = False
    ring[1, 26] = False

    stats_before = enclosed_hole_stats(ring, max_hole_area=MAX_SMALL_HOLE_AREA)
    assert stats_before["hole_count"] == 3
    assert stats_before["small_hole_count"] == 2

    filled = fill_small_enclosed_holes(ring, max_hole_area=MAX_SMALL_HOLE_AREA)
    assert filled.shape == ring.shape
    assert filled.dtype == bool
    assert filled[4, 4]  # diagonal ring pinhole filled
    assert filled[3:6, 9:14].all()  # 15-px hole filled
    assert not filled[3:7, 18:22].any()  # 16-px hole preserved!
    assert not filled[0, 26] and not filled[1, 26]  # border-touching bay NEVER filled!

    # Under 8-connected background, the diagonal 8-ring leaks outside and (4,4) stays unfilled.
    filled_bg8 = fill_small_enclosed_holes(ring, max_hole_area=MAX_SMALL_HOLE_AREA, connectivity=8)
    assert not filled_bg8[4, 4]


def test_ablation_morphology_variants_contracts_and_determinism() -> None:
    mask = np.zeros((28, 32), dtype=bool)
    # Dumbbell connected across a 1-px diagonal bridge so OPEN splits it into two <16 px pieces.
    mask[4:7, 4:8] = True
    mask[6, 8] = True
    mask[7:10, 9:13] = True
    # Large 6x6 block with a 1-px pinhole and 2-px vertical notch
    mask[14:22, 16:24] = True
    mask[17, 19] = False
    mask[18:20, 22] = False
    # 5-px isolated noise speck removed by M1
    mask[24, 2:7] = True

    res_a = ablation_morphology_variants(mask)
    res_b = ablation_morphology_variants(mask)
    base = morphology_variants(mask)

    assert tuple(res_a.keys()) == ALL_VARIANTS
    for var in ALL_VARIANTS:
        assert res_a[var].shape == mask.shape
        assert res_a[var].dtype == bool
        assert np.array_equal(res_a[var], res_b[var])
    for bvar in BASELINE_VARIANTS:
        assert np.array_equal(res_a[bvar], base[bvar])

    # Mathematical relationship: M2 == CLOSE(open_only)
    reclosed = (
        cv2.morphologyEx(
            res_a["open_only"].astype(np.uint8),
            cv2.MORPH_CLOSE,
            CROSS_3X3_KERNEL,
            iterations=1,
            borderType=cv2.BORDER_REPLICATE,
        )
        > 0
    )
    assert np.array_equal(res_a["M2"], reclosed)

    # m2_area16 must strictly have 0 components < 16 px while removing fragments created by M2/OPEN
    _, _, stats_m2, _ = cv2.connectedComponentsWithStats(res_a["M2"].astype(np.uint8), connectivity=8)
    _, _, stats_a16, _ = cv2.connectedComponentsWithStats(res_a["m2_area16"].astype(np.uint8), connectivity=8)
    if len(stats_a16) > 1:
        assert int(np.min(stats_a16[1:, cv2.CC_STAT_AREA])) >= MIN_COMPONENT_AREA
    assert int(np.sum(stats_m2[1:, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA)) >= 1
    assert res_a["hole_fill_small"][17, 19]  # interior pinhole closed without expanding outer perimeter


def test_validation_guards_raise_cleanly() -> None:
    bad_mask = np.zeros((4, 4, 3), dtype=bool)
    with pytest.raises(ValueError, match="2D binary mask"):
        filter_small_components(bad_mask)
    with pytest.raises(ValueError, match="2D binary mask"):
        fill_small_enclosed_holes(bad_mask)
    with pytest.raises(ValueError, match="min_area"):
        filter_small_components(np.zeros((8, 8), dtype=bool), min_area=0)
    with pytest.raises(ValueError, match="max_hole_area"):
        fill_small_enclosed_holes(np.zeros((8, 8), dtype=bool), max_hole_area=-1)
    with pytest.raises(ValueError, match="connectivity"):
        fill_small_enclosed_holes(np.zeros((8, 8), dtype=bool), connectivity=6)


def test_split_isolation_and_synthetic_cv_pipeline() -> None:
    manifest = load_split_manifest(REPO_ROOT, verify=True)
    # Even when full manifest includes test rows, build_strict_train_folds touches zero test/val rows.
    train_df, folds, oof_fold_ids = build_strict_train_folds(manifest, n_splits=CV_FOLDS)
    assert len(train_df) == 1535
    assert set(train_df["split"]) == {"train"}
    assert len(folds) == CV_FOLDS
    assert len(oof_fold_ids) == 1535
    for tr_idx, va_idx in folds:
        tr_groups = set(train_df.iloc[tr_idx]["group_id"])
        va_groups = set(train_df.iloc[va_idx]["group_id"])
        assert tr_groups.isdisjoint(va_groups)

    # Synthetic deterministic strict CV test across anchor specs + plots
    rng = np.random.default_rng(11)
    rows = []
    for i in range(36):
        rows.append(
            {
                "sample_id": f"Syn/S_{i:03d}",
                "category": EXPECTED_CATEGORIES[i % len(EXPECTED_CATEGORIES)],
                "group_id": f"G_{i // 3:02d}",
                "split": "train",
            }
        )
    syn_train_df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    _, syn_folds, _ = build_strict_train_folds(syn_train_df, n_splits=3)
    app_mat = rng.normal(0.0, 1.0, size=(len(syn_train_df), len(APPEARANCE_FEATURE_NAMES)))
    fold_morph = {
        fam: rng.normal(0.0, 1.0, size=(3, len(syn_train_df), len(ALL_VARIANTS), len(MORPHOLOGY_FEATURE_NAMES)))
        for fam in ("otsu", "k4")
    }
    cv1, f1, cat1, oof1 = evaluate_anchor_cv_grid(
        train_df=syn_train_df,
        folds=syn_folds,
        appearance_train=app_mat,
        family_fold_morph=fold_morph,
        anchor_specs=PREDECLARED_ANCHOR_SPECS[:2],
    )
    cv2, f2, _, _ = evaluate_anchor_cv_grid(
        train_df=syn_train_df,
        folds=syn_folds,
        appearance_train=app_mat,
        family_fold_morph=fold_morph,
        anchor_specs=PREDECLARED_ANCHOR_SPECS[:2],
    )
    pd.testing.assert_frame_equal(cv1, cv2)
    pd.testing.assert_frame_equal(f1, f2)
    assert len(cv1) == (1 + 2 * len(ALL_VARIANTS) * 2) * 2
    assert set(oof1["variant"]) == set(ALL_VARIANTS)
    assert set(cat1["category"]) == set(EXPECTED_CATEGORIES)

    shift_df = compute_morph_feature_shifts(
        {fam: fold_morph[fam][0] for fam in ("otsu", "k4")}
    )
    fig_s = plot_ablation_feature_shifts(shift_df, family="otsu")
    finish_figure(fig_s, show=False, close=True)
    fig_cv = plot_ablation_cv_macro_f1(cv1, f1)
    finish_figure(fig_cv, show=False, close=True)
    fig_cat = plot_ablation_per_category_f1(
        cat1, family="otsu", feature_set="C", spec_id="lin_none_C3"
    )
    finish_figure(fig_cat, show=False, close=True)


def test_saved_ablation_outputs_and_strict_v3_parity() -> None:
    out_dir = REPO_ROOT / "results/morphology_ablation"
    if not (out_dir / "ablation_manifest.json").is_file():
        pytest.skip("Saved morphology ablation results not yet generated.")

    outputs = load_saved_ablation_outputs(REPO_ROOT)
    assert outputs.manifest["test_images_touched"] == 0
    assert outputs.manifest["test_labels_accessed"] is False
    assert int(outputs.manifest["max_small_hole_area_predeclared"]) == MAX_SMALL_HOLE_AREA
    assert tuple(outputs.manifest["variants"]) == ALL_VARIANTS

    # Verify exact parity of M0/M1/M2 CV baselines against saved strict-v3 grid results!
    strict_grid = pd.read_csv(
        REPO_ROOT / CLASSIFICATION_2X3_STRICT_DIR_REL / "svm_2x3_strict_grid_results.csv"
    )
    strict_val = pd.read_csv(
        REPO_ROOT / CLASSIFICATION_2X3_STRICT_DIR_REL / "svm_2x3_strict_selected_summary.csv"
    )
    cv_df = outputs.cv_results

    for fam in ("otsu", "k4"):
        for var in BASELINE_VARIANTS:
            for fset in ("B", "C"):
                method_id = f"{fam}_{var}"
                sel_row = strict_val.loc[
                    strict_val["method"].eq(method_id)
                    & strict_val["feature_set"].eq(fset)
                    & strict_val["model_variant"].eq("cv_selected")
                ].iloc[0]
                match_ab = cv_df.loc[
                    cv_df["family"].eq(fam)
                    & cv_df["variant"].eq(var)
                    & cv_df["feature_set"].eq(fset)
                    & cv_df["kernel"].eq(sel_row["kernel"])
                    & cv_df["class_weight"].eq(sel_row["class_weight"])
                    & np.isclose(cv_df["C"], float(sel_row["C"]))
                    & cv_df["gamma"].astype(str).eq(str(sel_row["gamma"]))
                ]
                if not match_ab.empty:
                    np.testing.assert_allclose(
                        float(match_ab["cv_macro_f1"].iloc[0]),
                        float(sel_row["cv_macro_f1"]),
                        atol=1e-10,
                    )
                    np.testing.assert_allclose(
                        float(match_ab["val_macro_f1_readonly"].iloc[0]),
                        float(sel_row["val_macro_f1"]),
                        atol=1e-10,
                    )

    # Verify segmentation plot helpers render cleanly from saved dataframes
    for fig in (
        plot_ablation_segmentation_overview(outputs.seg_train_summary),
        plot_ablation_delta_dice_distributions(outputs.seg_train_per_image),
        plot_ablation_pixel_flips_and_components(outputs.seg_train_summary),
        plot_ablation_feature_shifts(outputs.feature_shifts, family="otsu"),
        plot_ablation_cv_macro_f1(outputs.cv_results, outputs.fold_deltas),
    ):
        finish_figure(fig, show=False, close=True)
