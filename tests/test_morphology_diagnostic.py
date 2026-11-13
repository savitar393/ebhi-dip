"""Tests for Task 1 morphological postprocessing diagnostic helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ebhi_dip.analysis import load_validation_results
from ebhi_dip.dataset import EXPECTED_CATEGORIES
from ebhi_dip.metrics import binary_segmentation_metrics
from ebhi_dip.morphology_diagnostic import (
    OUTCOME_LABELS,
    _bin_area_percentages,
    _load_case_bundle,
    classify_delta_outcomes,
    connected_component_areas,
    evaluate_val_morphology_cohort,
    file_sha256_map,
    load_development_morphology_csv,
    plot_case_morphology_progression,
    plot_component_distributions,
    plot_coverage_and_pixel_flips,
    plot_metric_distributions_and_tradeoffs,
    plot_representative_cases_grid,
    plot_seed_baseline_grid,
    render_component_highlight_rgb,
    render_rgb_with_gt_contour,
    render_transition_diff_rgb,
    select_deterministic_representative_cases,
    summarize_transitions_by_category,
)
from ebhi_dip.paths import find_project_root, has_raw_dataset
from ebhi_dip.plotting import finish_figure
from ebhi_dip.segmentation import morphology_variants

PROJECT_ROOT = find_project_root()


def test_transition_diff_is_disjoint_and_conserves_pixels() -> None:
    m0 = np.zeros((16, 16), dtype=bool)
    # 5x5 block (25 px >= 16) with a 1-px internal hole filled by M2 close.
    m0[2:7, 2:7] = True
    m0[4, 4] = False
    # 1-px thin protruding tip opened away by M2.
    m0[4, 7] = True
    # Isolated 3-px speckle removed by M1.
    m0[12, 12:15] = True

    stages = morphology_variants(m0)
    m1, m2 = stages["M1"], stages["M2"]
    diff_rgb, counts = render_transition_diff_rgb(m0, m1, m2)

    assert diff_rgb.shape == (16, 16, 3)
    assert diff_rgb.dtype == np.uint8
    assert counts["removed_m1_px"] == 3
    assert counts["removed_m2_px"] >= 1
    assert counts["added_m2_px"] >= 1
    assert int(np.sum(m1)) == counts["unchanged_px"] + counts["removed_m2_px"]
    assert int(np.sum(m2)) == counts["unchanged_px"] + counts["added_m2_px"]
    union_fg = m0 | m2
    assert (
        int(np.sum(union_fg))
        == counts["unchanged_px"]
        + counts["removed_m1_px"]
        + counts["removed_m2_px"]
        + counts["added_m2_px"]
    )


def test_refragmentation_and_component_area_helpers() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    # Dumbbell connected across diagonal 1-px waist (~13 px + 13 px) separated cleanly by OPEN.
    mask[3:6, 2:6] = True
    mask[5, 6] = True
    mask[6:9, 7:10] = True
    # Small 4-px island removed at M1.
    mask[15:17, 15:17] = True

    stages = morphology_variants(mask)
    _, n0, s0 = render_component_highlight_rgb(stages["M0"])
    _, n1, s1 = render_component_highlight_rgb(stages["M1"])
    _, n2, s2 = render_component_highlight_rgb(stages["M2"])

    assert n0 == 2 and s0 == 1
    assert n1 == 1 and s1 == 0
    assert n2 == 2 and s2 >= 1

    areas_m2 = connected_component_areas(stages["M2"])
    assert len(areas_m2) == 2
    assert bool(np.any(areas_m2 < 16))
    pcts = _bin_area_percentages(areas_m2)
    np.testing.assert_allclose(pcts.sum(), 100.0, atol=1e-12)
    np.testing.assert_allclose(_bin_area_percentages(np.empty(0, dtype=np.int32)), 0.0)


def test_outcome_buckets_and_test_split_guard() -> None:
    deltas = pd.Series([0.02, 0.005, 0.0049, 0.0, -0.0049, -0.005, -0.03])
    labels = classify_delta_outcomes(deltas, negligible_threshold=0.005).tolist()
    assert labels == [
        OUTCOME_LABELS[0],
        OUTCOME_LABELS[0],
        OUTCOME_LABELS[1],
        OUTCOME_LABELS[1],
        OUTCOME_LABELS[1],
        OUTCOME_LABELS[2],
        OUTCOME_LABELS[2],
    ]

    with pytest.raises(ValueError, match="Forbidden or unknown split"):
        evaluate_val_morphology_cohort(PROJECT_ROOT, split="test")

    dev_diag = load_development_morphology_csv(PROJECT_ROOT)
    assert set(dev_diag["split"].unique()) == {"train", "val"}


def test_synthetic_summaries_and_metric_plots() -> None:
    rows = []
    rng = np.random.default_rng(7)
    for fam in ("otsu", "k4"):
        for idx, cat in enumerate(EXPECTED_CATEGORIES):
            d0 = 0.70 + 0.01 * idx
            d1 = d0 + (0.004 if fam == "otsu" else -0.0002)
            d2 = d1 + (-0.010 if cat == "Polyp" else 0.006)
            rows.append(
                {
                    "sample_id": f"{cat}/S_{idx:03d}",
                    "category": cat,
                    "group_id": f"G_{idx:03d}",
                    "split": "val",
                    "family": fam,
                    "otsu_threshold": 100.0,
                    "ref_fg_px": 25000,
                    "ref_fg_pct": 50.0,
                    "fg_px_M0": 24000,
                    "fg_px_M1": 23500,
                    "fg_px_M2": 23600,
                    "fg_pct_M0": 48.0,
                    "fg_pct_M1": 47.0,
                    "fg_pct_M2": 47.2,
                    "cc_M0": 100,
                    "cc_M1": 30,
                    "cc_M2": 35,
                    "small_cc_M0": 70,
                    "small_cc_M1": 0,
                    "small_cc_M2": 8,
                    "dice_M0": d0,
                    "dice_M1": d1,
                    "dice_M2": d2,
                    "iou_M0": d0 - 0.1,
                    "iou_M1": d1 - 0.1,
                    "iou_M2": d2 - 0.1,
                    "prec_M0": 0.7,
                    "prec_M1": 0.72,
                    "prec_M2": 0.73,
                    "rec_M0": 0.7,
                    "rec_M1": 0.695,
                    "rec_M2": 0.71 if cat != "Polyp" else 0.67,
                    "delta_dice_M1_M0": d1 - d0,
                    "delta_dice_M2_M1": d2 - d1,
                    "delta_dice_M2_M0": d2 - d0,
                    "delta_iou_M1_M0": d1 - d0,
                    "delta_iou_M2_M1": d2 - d1,
                    "delta_iou_M2_M0": d2 - d0,
                    "delta_prec_M1_M0": 0.02,
                    "delta_prec_M2_M1": 0.01,
                    "delta_rec_M1_M0": -0.005,
                    "delta_rec_M2_M1": 0.015 if cat != "Polyp" else -0.025,
                    "m1_rem_tp": 100,
                    "m1_rem_fp": 400,
                    "open_rem_tp": 800,
                    "open_rem_fp": 600,
                    "close_add_tp": 900,
                    "close_add_fp": 350,
                    "m02_rem_tp": 850,
                    "m02_rem_fp": 950,
                    "m02_add_tp": 850,
                    "m02_add_fp": 320,
                }
            )
    syn_df = pd.DataFrame(rows)
    rep = select_deterministic_representative_cases(syn_df)
    assert len(rep) == 7

    summary = summarize_transitions_by_category(syn_df)
    assert len(summary) == 14
    assert set(summary["category"].unique()) == {*EXPECTED_CATEGORIES, "Macro (6 cat)"}

    areas = {
        (fam, st): rng.integers(1, 1500, size=40, dtype=np.int32)
        for fam in ("otsu", "k4")
        for st in ("M0", "M1", "M2")
    }
    dev_diag = load_development_morphology_csv(PROJECT_ROOT)
    for fig in (
        plot_component_distributions(areas, syn_df, dev_diag),
        plot_coverage_and_pixel_flips(syn_df),
        plot_metric_distributions_and_tradeoffs(syn_df),
    ):
        finish_figure(fig, show=False, close=True)


def test_seed_cases_match_saved_val_csv_and_render() -> None:
    if not has_raw_dataset(PROJECT_ROOT):
        pytest.skip("Raw EBHI-SEG dataset not present.")

    seed_ids = (
        "Adenocarcinoma/GTXC2110135-2-400-008",
        "Adenocarcinoma/GTxc2013489-1-400-006",
    )
    saved_val = load_validation_results(PROJECT_ROOT)
    hashes = file_sha256_map(PROJECT_ROOT, ["configs/otsu_raw.json", "results/otsu_morph_val.csv"])
    assert len(hashes) == 2

    for sid in seed_ids:
        rec, rgb, ref, _, states = _load_case_bundle(PROJECT_ROOT, sid)
        assert rec["split"] == "val"
        contoured = render_rgb_with_gt_contour(rgb, ref)
        assert contoured.shape == rgb.shape

        for fam in ("otsu", "k4"):
            for st in ("M0", "M1", "M2"):
                met = binary_segmentation_metrics(states[fam][st], ref)
                ref_row = saved_val.loc[
                    saved_val["sample_id"].eq(sid) & saved_val["method"].eq(f"{fam}_{st}")
                ].iloc[0]
                np.testing.assert_allclose(met["dice"], float(ref_row["dice"]), atol=1e-10)
                np.testing.assert_allclose(met["iou"], float(ref_row["iou"]), atol=1e-10)

    fig_base = plot_seed_baseline_grid(PROJECT_ROOT, seed_ids)
    finish_figure(fig_base, show=False, close=True)
    fig_prog = plot_case_morphology_progression(PROJECT_ROOT, seed_ids[0])
    finish_figure(fig_prog, show=False, close=True)
    fig_rep = plot_representative_cases_grid(
        PROJECT_ROOT,
        [("Seed 1", "otsu", seed_ids[0]), ("Seed 2", "k4", seed_ids[1])],
    )
    finish_figure(fig_rep, show=False, close=True)
