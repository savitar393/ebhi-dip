"""Tests for paired case ranking, deduplication, histogram pixel conservation, and CSV parity."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ebhi_dip.analysis import (
    build_case,
    load_validation_results,
    pair_scores,
    paired_outcome_counts,
    rank_cases,
    select_distinct_cases,
)
from ebhi_dip.dataset import load_split_manifest
from ebhi_dip.paths import has_raw_dataset
from ebhi_dip.plotting import (
    build_error_overlay,
    case_metrics_table,
    compute_case_histograms,
    compute_histogram_1d,
    compute_region_histograms,
    compute_shared_lab_edges,
    format_case_metrics_for_display,
    plot_image_histograms,
    plot_paired_scatter,
    plot_region_lab_histograms,
    plot_score_histograms,
    plot_segmentation_case,
    uint8_bin_edges,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_acceptance_ranking_fixture_and_deduplication() -> None:
    fixture = pd.DataFrame(
        [
            {"sample_id": "s1", "category": "Normal", "group_id": "G1", "method": "otsu_M0", "dice": 0.1, "iou": 0.05},
            {"sample_id": "s1", "category": "Normal", "group_id": "G1", "method": "k4_M0", "dice": 0.8, "iou": 0.67},
            {"sample_id": "s2", "category": "Polyp", "group_id": "G2", "method": "otsu_M0", "dice": 0.9, "iou": 0.82},
            {"sample_id": "s2", "category": "Polyp", "group_id": "G2", "method": "k4_M0", "dice": 0.2, "iou": 0.11},
            {"sample_id": "s3", "category": "Normal", "group_id": "G3", "method": "otsu_M0", "dice": 0.4, "iou": 0.25},
            {"sample_id": "s3", "category": "Normal", "group_id": "G3", "method": "k4_M0", "dice": 0.4, "iou": 0.25},
        ]
    )
    paired = pair_scores(fixture, "otsu_M0", "k4_M0")

    assert rank_cases(paired, "worst_a", n=1)["sample_id"].tolist() == ["s1"]
    assert rank_cases(paired, "worst_b", n=1)["sample_id"].tolist() == ["s2"]
    assert rank_cases(paired, "a_advantage", n=3)["sample_id"].tolist() == ["s2"]
    assert rank_cases(paired, "b_advantage", n=3)["sample_id"].tolist() == ["s1"]
    assert rank_cases(paired, "both_poor", n=1)["sample_id"].tolist() == ["s3"]
    assert rank_cases(paired, "ties", n=3)["sample_id"].tolist() == ["s3"]
    assert rank_cases(paired, "best_a", n=1)["sample_id"].tolist() == ["s2"]
    assert rank_cases(paired, "best_b", n=1)["sample_id"].tolist() == ["s1"]

    deduped = select_distinct_cases(paired)
    assert deduped["sample_id"].tolist() == ["s1", "s2"]
    assert deduped.loc[deduped["sample_id"].eq("s1"), "selection_modes"].iloc[0] == (
        "worst_a",
        "b_advantage",
    )
    assert deduped.loc[deduped["sample_id"].eq("s2"), "selection_modes"].iloc[0] == (
        "worst_b",
        "a_advantage",
    )


def test_pairing_errors_and_empty_directional_modes() -> None:
    all_b_better = pd.DataFrame(
        [
            {"sample_id": "s2", "category": "Normal", "group_id": "G2", "method": "A", "dice": 0.3, "iou": 0.2},
            {"sample_id": "s2", "category": "Normal", "group_id": "G2", "method": "B", "dice": 0.6, "iou": 0.4},
            {"sample_id": "s1", "category": "Normal", "group_id": "G1", "method": "A", "dice": 0.3, "iou": 0.2},
            {"sample_id": "s1", "category": "Normal", "group_id": "G1", "method": "B", "dice": 0.6, "iou": 0.4},
        ]
    )
    paired = pair_scores(all_b_better, "A", "B")
    assert rank_cases(paired, "a_advantage", n=3).empty
    assert rank_cases(paired, "ties", n=3).empty
    # Deterministic sample_id ascending tie-break: s1 comes before s2!
    assert rank_cases(paired, "worst_a", n=2)["sample_id"].tolist() == ["s1", "s2"]

    unpaired = all_b_better.iloc[:-1].copy()
    with pytest.raises(ValueError, match="Unpaired sample_id mismatch"):
        pair_scores(unpaired, "A", "B")

    dup = pd.concat([all_b_better, all_b_better.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        pair_scores(dup, "A", "B")

    cat_mismatch = all_b_better.copy()
    cat_mismatch.loc[0, "category"] = "Polyp"
    with pytest.raises(ValueError, match="Cohort 'category' mismatch"):
        pair_scores(cat_mismatch, "A", "B")

    test_split = all_b_better.copy()
    test_split["split"] = "test"
    with pytest.raises(ValueError, match="only permits validation"):
        pair_scores(test_split, "A", "B")


def test_validation_paired_counts_from_csv() -> None:
    results = load_validation_results(REPO_ROOT)

    raw_pair = pair_scores(results, "otsu_M0", "k4_M0")
    raw_counts, raw_by_cat = paired_outcome_counts(raw_pair)
    assert raw_counts == {"a_higher": 96, "b_higher": 276, "tie": 0, "total": 372}
    assert int(raw_by_cat["total"].sum()) == 372

    proc_pair = pair_scores(results, "otsu_M2", "k4_M1")
    proc_counts, _ = paired_outcome_counts(proc_pair)
    assert proc_counts == {"a_higher": 122, "b_higher": 250, "tie": 0, "total": 372}


def test_histogram_pixel_conservation_signed_lab_constant_and_empty_region() -> None:
    # 1. Extreme uint8 values including 0 and 255 must all be counted.
    u8_vals = np.array([0, 0, 1, 127, 128, 254, 255, 255], dtype=np.uint8)
    for b_count in (16, 64, 256):
        h_u8 = compute_histogram_1d(u8_vals, uint8_bin_edges(b_count))
        assert int(h_u8["counts"].sum()) == u8_vals.size

    # 2. Signed physical Lab including values beyond +/-128 and constant channel!
    lab_a = np.zeros((10, 10, 3), dtype=np.float32)
    lab_a[:, :, 0] = 55.0  # Constant L* channel!
    lab_a[:5, :, 1] = -142.5  # Signed negative beyond -128
    lab_a[5:, :, 1] = 138.25  # Signed positive beyond +128
    lab_a[:, :, 2] = np.linspace(-60.0, 45.0, 100, dtype=np.float32).reshape(10, 10)

    shared_edges = compute_shared_lab_edges([lab_a], bins=32)
    for idx, ch in enumerate(("L*", "a*", "b*")):
        h_cnt = compute_histogram_1d(lab_a[:, :, idx], shared_edges[ch], density=False)
        assert int(h_cnt["counts"].sum()) == 100
        h_den = compute_histogram_1d(lab_a[:, :, idx], shared_edges[ch], density=True)
        integral = float(np.sum(h_den["heights"] * np.diff(h_den["edges"])))
        assert integral == pytest.approx(1.0, abs=1e-12)

    # 3. Empty region handling in both count and density modes (must never produce NaN).
    empty_mask = np.zeros((10, 10), dtype=bool)
    full_mask = np.ones((10, 10), dtype=bool)
    for den_flag in (False, True):
        reg = compute_region_histograms(
            lab_a[:, :, 1],
            {"empty_fg": empty_mask, "full_fg": full_mask},
            edges=shared_edges["a*"],
            density=den_flag,
        )
        assert reg["empty_fg"]["empty"] is True
        assert reg["empty_fg"]["pixel_count"] == 0
        assert int(reg["empty_fg"]["counts"].sum()) == 0
        assert np.isfinite(reg["empty_fg"]["heights"]).all()
        assert reg["full_fg"]["pixel_count"] == 100


def test_error_overlay_and_selected_validation_cases_parity() -> None:
    rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
    pred = np.array([[True, True], [False, False]])
    ref = np.array([[True, False], [True, False]])
    overlay, counts = build_error_overlay(rgb, pred, ref)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert np.array_equal(overlay[1, 1], np.array([100, 100, 100], dtype=np.uint8))  # TN unchanged!
    assert not np.array_equal(overlay[0, 0], overlay[0, 1])

    if not has_raw_dataset(REPO_ROOT):
        pytest.skip("Raw EBHI-SEG dataset not present in checkout.")

    results = load_validation_results(REPO_ROOT)
    paired = pair_scores(results, "otsu_M0", "k4_M0")
    selected = select_distinct_cases(paired)
    assert 1 <= len(selected) <= 4

    split_df = load_split_manifest(REPO_ROOT)
    cases = [
        build_case(
            REPO_ROOT,
            row,
            "otsu_M0",
            "k4_M0",
            selection_reasons=row["selection_reasons"],
            split_df=split_df,
        )
        for _, row in selected.iterrows()
    ]

    for case in cases:
        hists = compute_case_histograms(case, bins=64)
        for key in ("R", "G", "B", "gray", "reference_raw", "L*", "a*", "b*"):
            assert int(hists[key]["counts"].sum()) == 224 * 224

    fig_case = plot_segmentation_case(cases[0])
    plt.close(fig_case)
    fig_hist = plot_image_histograms(cases[0], bins=64)
    plt.close(fig_hist)
    fig_reg = plot_region_lab_histograms(cases[0], bins=48)
    plt.close(fig_reg)
    fig_scatter = plot_paired_scatter(paired)
    plt.close(fig_scatter)
    fig_scores = plot_score_histograms(paired, bins=20)
    plt.close(fig_scores)

    tbl = format_case_metrics_for_display(case_metrics_table(cases[0]))
    assert list(tbl.index) == ["otsu_M0", "k4_M0"]
