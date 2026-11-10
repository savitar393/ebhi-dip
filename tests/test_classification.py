"""Tests for Stage 07 six-class SVM classification, scaler/model parity, and split discipline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ebhi_dip.classification import (
    ALL_FEATURE_SETS,
    DECISION_PARITY_ATOL,
    PRIMARY_FEATURE_SETS,
    apply_standard_scaler,
    classification_metrics,
    confusion_matrix_table,
    export_svm_artifact,
    feature_columns_for_set,
    fit_standard_scaler,
    fit_svm_model,
    load_classification_feature_frame,
    load_svm_artifact,
    per_category_classification_table,
    predict_from_svm_artifact,
    selection_key_for_svm,
    verify_svm_artifact_parity,
)
from ebhi_dip.dataset import EXPECTED_CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_feature_columns_for_set_contracts() -> None:
    assert len(feature_columns_for_set("A")) == 10
    assert len(feature_columns_for_set("B")) == 19
    assert len(feature_columns_for_set("C")) == 29
    assert len(feature_columns_for_set("A_seg")) == 10
    assert len(feature_columns_for_set("C_seg")) == 29
    with pytest.raises(ValueError, match="Unknown feature_set"):
        feature_columns_for_set("invalid_set")


def test_standard_scaler_matches_sklearn_and_handles_zero_variance() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(loc=10.0, scale=3.5, size=(30, 4))
    X[:, 2] = 7.0  # constant column
    names = ("f0", "f1", "f2_const", "f3")

    payload = fit_standard_scaler(X, names)
    Z_our = apply_standard_scaler(X, payload)

    sk_scaler = StandardScaler()
    Z_sk = sk_scaler.fit_transform(X)
    np.testing.assert_allclose(Z_our, Z_sk, rtol=0.0, atol=1e-12)
    assert np.allclose(Z_our[:, 2], 0.0)


def test_classification_metrics_and_per_category_tables() -> None:
    y_true = list(EXPECTED_CATEGORIES) * 2
    y_pred = list(EXPECTED_CATEGORIES) * 2
    perfect = classification_metrics(y_true, y_pred)
    assert perfect["macro_f1"] == pytest.approx(1.0)
    assert perfect["balanced_accuracy"] == pytest.approx(1.0)
    assert perfect["accuracy"] == pytest.approx(1.0)

    cat_table = per_category_classification_table(y_true, y_pred)
    assert list(cat_table["category"]) == list(EXPECTED_CATEGORIES)
    assert (cat_table["support"] == 2).all()
    assert (cat_table["tp"] == 2).all()
    assert (cat_table["fp"] == 0).all()
    assert (cat_table["fn"] == 0).all()

    cm = confusion_matrix_table(y_true, y_pred)
    assert cm.shape == (6, 6)
    assert np.diag(cm.to_numpy()).sum() == len(y_true)


def test_selection_key_tie_breaking_order() -> None:
    row_rbf = {
        "val_macro_f1": 0.35001,
        "val_balanced_accuracy": 0.34001,
        "val_accuracy": 0.52001,
        "cv_macro_f1": 0.36001,
        "kernel": "rbf",
        "C": 10.0,
    }
    row_linear = {**row_rbf, "kernel": "linear"}
    row_smaller_c = {**row_rbf, "C": 1.0}

    assert selection_key_for_svm(row_linear) > selection_key_for_svm(row_rbf)
    assert selection_key_for_svm(row_smaller_c) > selection_key_for_svm(row_rbf)


def test_load_classification_feature_frame_excludes_test() -> None:
    train_df, val_df, split_sha = load_classification_feature_frame(
        REPO_ROOT, method="k4_M0", segmented=False
    )
    assert len(train_df) == 1535
    assert len(val_df) == 372
    assert set(train_df["split"]) == {"train"}
    assert set(val_df["split"]) == {"val"}
    assert set(train_df["sample_id"]).isdisjoint(set(val_df["sample_id"]))
    assert set(train_df["group_id"]).isdisjoint(set(val_df["group_id"]))
    assert len(split_sha) == 64


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_export_and_predict_svm_artifact_parity(kernel: str) -> None:
    train_df, val_df, split_sha = load_classification_feature_frame(
        REPO_ROOT, method="k4_M0", segmented=False
    )
    cols = feature_columns_for_set("A")
    X_tr_raw = train_df[list(cols)].to_numpy(dtype=np.float64)
    y_tr = train_df["category"].to_numpy(dtype=object)
    X_va_raw = val_df[list(cols)].to_numpy(dtype=np.float64)
    y_va = val_df["category"].to_numpy(dtype=object)

    scaler_payload = fit_standard_scaler(X_tr_raw, cols)
    X_tr = apply_standard_scaler(X_tr_raw, scaler_payload)
    clf = fit_svm_model(X_tr, y_tr, kernel=kernel, C=1.0, gamma="scale", class_weight="none")

    X_va = apply_standard_scaler(X_va_raw, scaler_payload)
    metrics = classification_metrics(y_va, clf.predict(X_va))
    artifact = export_svm_artifact(
        model_id=f"test_{kernel}",
        segmentation_method="k4_M0",
        feature_set="A",
        scaler_payload=scaler_payload,
        clf=clf,
        split_sha256=split_sha,
        validation_metrics=metrics,
    )
    parity = verify_svm_artifact_parity(X_va_raw, clf, artifact, atol=DECISION_PARITY_ATOL)
    assert parity["exact_prediction_match"] is True
    assert float(parity["max_abs_decision_diff"]) <= DECISION_PARITY_ATOL


def test_saved_stage07_artifacts_integrity_and_parity() -> None:
    cls_dir = REPO_ROOT / "results/classification"
    manifest_path = cls_dir / "classification_manifest.json"
    assert manifest_path.is_file()

    grid_df = pd.read_csv(cls_dir / "svm_grid_results.csv")
    summary_df = pd.read_csv(cls_dir / "svm_selected_summary.csv")
    per_cat_df = pd.read_csv(cls_dir / "svm_per_category_val.csv")
    comp_df = pd.read_csv(cls_dir / "svm_comparator_sensitivity.csv")
    pred_df = pd.read_csv(cls_dir / "svm_val_predictions.csv")

    assert set(grid_df["feature_set"]) == set(ALL_FEATURE_SETS)
    assert set(summary_df["feature_set"]) == set(ALL_FEATURE_SETS)
    assert set(per_cat_df["category"]) == set(EXPECTED_CATEGORIES)
    assert set(comp_df["method"]) == {"otsu_M0", "otsu_M2", "k4_M0", "k4_M1"}
    assert set(pred_df["split"]) == {"val"}

    _, val_df, _ = load_classification_feature_frame(REPO_ROOT, method="k4_M0", segmented=False)
    for set_name in PRIMARY_FEATURE_SETS:
        model_path = REPO_ROOT / "models" / f"svm_k4_M0_{set_name}.json"
        artifact = load_svm_artifact(model_path)
        cols = feature_columns_for_set(set_name)
        preds, _, _ = predict_from_svm_artifact(
            val_df[list(cols)].to_numpy(dtype=np.float64), artifact
        )
        stored_preds = pred_df.loc[
            pred_df["method"].eq("k4_M0")
            & pred_df["feature_set"].eq(set_name)
            & pred_df["model_variant"].eq("selected_best")
        ].sort_values("sample_id")
        assert np.array_equal(preds, stored_preds["predicted_category"].to_numpy(dtype=object))
