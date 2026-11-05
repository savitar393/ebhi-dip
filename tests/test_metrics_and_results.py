"""Tests for overlap metrics, confusion counts, category macro aggregation, and validation harmonization."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ebhi_dip.analysis import (
    load_validation_results,
    selection_ranking_key,
    summarize_validation_methods,
)
from ebhi_dip.metrics import (
    binary_segmentation_metrics,
    category_macro_scores,
    confusion_counts,
    overlap_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_overlap_scores_and_confusion_fixture() -> None:
    pred = [[1, 1], [0, 0]]
    ref = [[1, 0], [1, 0]]
    dice, iou = overlap_scores(pred, ref)
    assert dice == pytest.approx(0.5, abs=1e-15)
    assert iou == pytest.approx(1.0 / 3.0, abs=1e-15)

    counts = confusion_counts(pred, ref)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}

    metrics = binary_segmentation_metrics(pred, ref)
    assert metrics["precision"] == pytest.approx(0.5, abs=1e-15)
    assert metrics["recall"] == pytest.approx(0.5, abs=1e-15)


def test_overlap_and_zero_denominator_edge_cases() -> None:
    empty = np.zeros((3, 3), dtype=bool)
    full = np.ones((3, 3), dtype=bool)

    assert overlap_scores(empty, empty) == (1.0, 1.0)
    assert overlap_scores(full, full) == (1.0, 1.0)
    assert overlap_scores(empty, full) == (0.0, 0.0)
    assert overlap_scores(full, empty) == (0.0, 0.0)

    both_empty_metrics = binary_segmentation_metrics(empty, empty)
    assert both_empty_metrics["dice"] == 1.0
    assert both_empty_metrics["iou"] == 1.0
    assert math.isnan(float(both_empty_metrics["precision"]))
    assert math.isnan(float(both_empty_metrics["recall"]))

    pred_empty_metrics = binary_segmentation_metrics(empty, full)
    assert math.isnan(float(pred_empty_metrics["precision"]))
    assert pred_empty_metrics["recall"] == 0.0

    ref_empty_metrics = binary_segmentation_metrics(full, empty)
    assert ref_empty_metrics["precision"] == 0.0
    assert math.isnan(float(ref_empty_metrics["recall"]))

    with pytest.raises(ValueError, match="shape mismatch"):
        overlap_scores(np.zeros((2, 2), dtype=bool), np.zeros((2, 3), dtype=bool))

    with pytest.raises(ValueError, match="outside binary"):
        overlap_scores([[0, 2]], [[0, 1]])


def test_category_macro_scores_unequal_category_sizes_and_guards() -> None:
    df = pd.DataFrame(
        [
            {"sample_id": "s1", "method": "m1", "category": "CatA", "dice": 1.0, "iou": 1.0},
            {"sample_id": "s2", "method": "m1", "category": "CatA", "dice": 1.0, "iou": 1.0},
            {"sample_id": "s3", "method": "m1", "category": "CatB", "dice": 0.0, "iou": 0.0},
        ]
    )
    macro = category_macro_scores(df, expected_categories=["CatA", "CatB"])
    assert macro.loc["m1", "mean_category_dice"] == pytest.approx(0.5, abs=1e-15)
    assert macro.loc["m1", "mean_category_iou"] == pytest.approx(0.5, abs=1e-15)

    with pytest.raises(ValueError, match="Category mismatch"):
        category_macro_scores(df, expected_categories=["CatA", "CatB", "CatC"])

    bad_range = df.copy()
    bad_range.loc[0, "dice"] = 1.2
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        category_macro_scores(bad_range)

    bad_nan = df.copy()
    bad_nan.loc[0, "iou"] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        category_macro_scores(bad_nan)

    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        category_macro_scores(dup)


def test_load_validation_results_checkpoint_reconciliation() -> None:
    results = load_validation_results(REPO_ROOT)
    assert len(results) == 4836
    assert results["method"].nunique() == 13
    assert results["sample_id"].nunique() == 372

    summary = summarize_validation_methods(results)
    expected_rounded = {
        "otsu_M0": (0.7232, 0.5788),
        "otsu_M2": (0.7288, 0.5886),
        "k4_M0": (0.7761, 0.6427),
        "k4_M1": (0.7760, 0.6426),
        "all_foreground": (0.7767, 0.6477),
    }
    for method_id, (exp_dice, exp_iou) in expected_rounded.items():
        assert round(float(summary.loc[method_id, "mean_category_dice"]), 4) == exp_dice
        assert round(float(summary.loc[method_id, "mean_category_iou"]), 4) == exp_iou

    candidates = [m for m in summary.index if m != "all_foreground"]
    best = max(candidates, key=lambda m: selection_ranking_key(summary, m))
    assert best == "k4_M0"
