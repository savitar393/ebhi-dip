"""Tests for Stage 08 held-out test evaluation, pre-test selection lock, and frozen model discipline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ebhi_dip.classification import (
    classification_metrics,
    feature_columns_for_set,
    load_svm_artifact,
    predict_from_svm_artifact,
)
from ebhi_dip.dataset import EXPECTED_CATEGORIES
from ebhi_dip.evaluation import (
    SEGMENTATION_SELECTION_REL,
    TEST_RESULTS_DIR_REL,
    build_segmentation_selection_config,
    load_segmentation_selection_config,
    load_test_feature_frame,
)
from ebhi_dip.segmentation import SUPPORTED_METHODS

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_segmentation_selection_config_contract() -> None:
    cfg = load_segmentation_selection_config(REPO_ROOT)
    recomputed = build_segmentation_selection_config(REPO_ROOT)
    assert cfg == recomputed
    assert cfg["primary_method"] == "k4_M0"
    assert cfg["selection_split"] == "val"
    assert cfg["cohort_counts"]["test_images_held_out"] == 319
    assert cfg["cohort_counts"]["test_groups_held_out"] == 53


def test_load_test_feature_frame_strictly_isolates_test() -> None:
    test_df, split_sha = load_test_feature_frame(REPO_ROOT, method="k4_M0", segmented=False)
    assert len(test_df) == 319
    assert int(test_df["group_id"].nunique()) == 53
    assert set(test_df["split"]) == {"test"}
    assert set(test_df["category"]) == set(EXPECTED_CATEGORIES)
    assert len(split_sha) == 64


def test_saved_stage08_test_artifacts_integrity_and_parity() -> None:
    test_dir = REPO_ROOT / TEST_RESULTS_DIR_REL
    manifest_path = test_dir / "final_test_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["evaluation_split"] == "test"
    assert manifest["cohort_counts"]["test_images"] == 319
    assert manifest["cohort_counts"]["test_groups"] == 53

    seg_img = pd.read_csv(test_dir / "segmentation_test_per_image.csv")
    seg_macro = pd.read_csv(test_dir / "segmentation_test_macro_summary.csv")
    seg_cat = pd.read_csv(test_dir / "segmentation_test_per_category.csv")
    seg_con = pd.read_csv(test_dir / "segmentation_test_paired_contrasts.csv")

    assert len(seg_img) == 319 * len(SUPPORTED_METHODS)
    assert set(seg_img["split"]) == {"test"}
    assert set(seg_macro["method"]) == set(SUPPORTED_METHODS)
    assert len(seg_cat) == len(SUPPORTED_METHODS) * len(EXPECTED_CATEGORIES)
    assert len(seg_con) == 6

    # Verify k4_M0 exceeds all_foreground control on held-out test
    seg_idx = seg_macro.set_index("method")
    assert float(seg_idx.loc["k4_M0", "test_mean_category_dice"]) > float(
        seg_idx.loc["all_foreground", "test_mean_category_dice"]
    )
    assert float(seg_idx.loc["k4_M0", "test_delta_dice_vs_control"]) > 0.0

    cls_sum = pd.read_csv(test_dir / "classification_test_summary.csv")
    cls_cat = pd.read_csv(test_dir / "classification_test_per_category.csv")
    cls_con = pd.read_csv(test_dir / "classification_test_paired_contrasts.csv")
    cls_pred = pd.read_csv(test_dir / "classification_test_predictions.csv")

    assert len(cls_sum) == 12
    assert len(cls_cat) == 12 * len(EXPECTED_CATEGORIES)
    assert not cls_cat[["val_recall", "val_f1", "delta_f1_test_minus_val"]].isna().any().any()
    assert len(cls_con) == 6
    assert set(cls_pred["split"]) == {"test"}

    # Verify pure-NumPy artifact inference on test matches stored predictions and summary metrics
    cls_sum_idx = cls_sum.set_index("model_id")
    for model_id, method_id, set_name in (
        ("svm_k4_M0_A", "k4_M0", "A"),
        ("svm_k4_M0_B", "k4_M0", "B"),
        ("svm_k4_M0_selected", "k4_M0", "C"),
        ("svm_otsu_M2_C", "otsu_M2", "C"),
    ):
        art = load_svm_artifact(REPO_ROOT / "models" / f"{model_id}.json")
        assert int(art["scaler"]["n_samples_seen"]) == 1535
        t_df, _ = load_test_feature_frame(REPO_ROOT, method=method_id, segmented=False)
        cols = feature_columns_for_set(set_name)
        preds, _, _ = predict_from_svm_artifact(t_df[list(cols)].to_numpy(dtype=np.float64), art)

        stored_sub = (
            cls_pred.loc[cls_pred["model_id"].eq(model_id)]
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        assert np.array_equal(preds, stored_sub["predicted_category"].to_numpy(dtype=object))
        m_scores = classification_metrics(t_df["category"], preds)
        assert abs(m_scores["macro_f1"] - float(cls_sum_idx.loc[model_id, "test_macro_f1"])) < 1e-12
        assert abs(m_scores["accuracy"] - float(cls_sum_idx.loc[model_id, "test_accuracy"])) < 1e-12
