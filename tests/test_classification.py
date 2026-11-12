"""Tests for Stage 07 six-class SVM classification, scaler/model parity, and split discipline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ebhi_dip.classification import (
    ALL_FEATURE_SETS,
    CLASSIFICATION_2X3_DIR_REL,
    DECISION_PARITY_ATOL,
    MODELS_CV_2X3_DIR_REL,
    PRIMARY_FEATURE_SETS,
    SHARED_APPEARANCE_METHOD,
    apply_standard_scaler,
    classification_metrics,
    confusion_matrix_table,
    cv_selection_key_for_svm,
    export_svm_artifact,
    feature_columns_for_set,
    fit_standard_scaler,
    fit_svm_model,
    load_classification_feature_frame,
    load_shared_appearance_frame,
    load_svm_artifact,
    per_category_classification_table,
    predict_from_svm_artifact,
    selection_key_for_svm,
    verify_svm_artifact_parity,
)
from ebhi_dip.dataset import EXPECTED_CATEGORIES
from ebhi_dip.features import APPEARANCE_FEATURE_NAMES, COMPARISON_2X3_METHODS
from ebhi_dip.provenance import sha256_bytes
from ebhi_dip.strict_cv import (
    CLASSIFICATION_2X3_STRICT_DIR_REL,
    MODELS_CV_2X3_STRICT_DIR_REL,
    compute_val_bootstrap_models_and_contrasts,
    evaluate_strict_cv_grid,
    fit_fold_kmeans_k4_mapped,
    fit_fold_otsu_polarity,
)

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


def test_cv_selection_key_ignores_val_metrics() -> None:
    high_cv_low_val = {
        "cv_macro_f1": 0.4500,
        "cv_balanced_accuracy": 0.4400,
        "cv_accuracy": 0.5800,
        "cv_fold_macro_f1_mean": 0.4450,
        "val_macro_f1": 0.1000,
        "val_balanced_accuracy": 0.1000,
        "val_accuracy": 0.1000,
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "0.01",
        "class_weight": "balanced",
    }
    low_cv_high_val = {
        "cv_macro_f1": 0.4000,
        "cv_balanced_accuracy": 0.4000,
        "cv_accuracy": 0.5500,
        "cv_fold_macro_f1_mean": 0.4000,
        "val_macro_f1": 0.9900,
        "val_balanced_accuracy": 0.9900,
        "val_accuracy": 0.9900,
        "kernel": "linear",
        "C": 0.1,
        "gamma": "scale",
        "class_weight": "none",
    }
    assert cv_selection_key_for_svm(high_cv_low_val) > cv_selection_key_for_svm(low_cv_high_val)

    tie_linear = {**high_cv_low_val, "kernel": "linear"}
    tie_smaller_c = {**high_cv_low_val, "C": 1.0}
    assert cv_selection_key_for_svm(tie_linear) > cv_selection_key_for_svm(high_cv_low_val)
    assert cv_selection_key_for_svm(tie_smaller_c) > cv_selection_key_for_svm(high_cv_low_val)


def test_load_shared_appearance_frame_excludes_test() -> None:
    train_app, val_app, split_sha = load_shared_appearance_frame(REPO_ROOT)
    assert len(train_app) == 1535
    assert len(val_app) == 372
    assert set(train_app["split"]) == {"train"}
    assert set(val_app["split"]) == {"val"}
    assert set(train_app["sample_id"]).isdisjoint(set(val_app["sample_id"]))
    assert set(train_app["group_id"]).isdisjoint(set(val_app["group_id"]))
    assert all(col in train_app.columns for col in APPEARANCE_FEATURE_NAMES)
    assert len(split_sha) == 64


def test_saved_stage07_2x3_artifacts_leakfree_and_parity() -> None:
    cls_2x3_dir = REPO_ROOT / CLASSIFICATION_2X3_DIR_REL
    manifest_path = cls_2x3_dir / "classification_2x3_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "svm-classification-2x3-v2"
    assert tuple(manifest["methods_2x3"]) == COMPARISON_2X3_METHODS
    assert manifest["cohort_counts"]["test_images_touched"] == 0
    assert manifest["cohort_counts"]["test_labels_accessed"] is False
    assert manifest["input_artifact_sha256"]["legacy_classification_manifest_v1"] == sha256_bytes(
        REPO_ROOT / "results/classification/classification_manifest.json"
    )

    grid_df = pd.read_csv(cls_2x3_dir / "svm_2x3_grid_results.csv")
    summary_df = pd.read_csv(cls_2x3_dir / "svm_2x3_selected_summary.csv")
    per_cat_df = pd.read_csv(cls_2x3_dir / "svm_2x3_per_category_val.csv")
    seg_cls_df = pd.read_csv(cls_2x3_dir / "svm_2x3_segmentation_vs_classification.csv")
    pred_df = pd.read_csv(cls_2x3_dir / "svm_2x3_val_predictions.csv")

    assert len(grid_df) == 15 * 32
    assert len(summary_df) == 15 * 2
    assert set(per_cat_df["category"]) == set(EXPECTED_CATEGORIES)
    assert set(pred_df["split"]) == {"val"}
    assert len(seg_cls_df) == 12
    assert set(seg_cls_df["method"]) == set(COMPARISON_2X3_METHODS)
    assert set(seg_cls_df["feature_set"]) == {"B", "C"}

    # Assert hyperparameter selection for every combination is invariant to masking val_* columns
    for (_, _), grp in grid_df.groupby(["method", "feature_set"], sort=False):
        sel = grp.loc[grp["is_selected_best"]]
        assert len(sel) == 1
        sel_row = sel.iloc[0]
        masked_rows = [
            {k: (-999.0 if str(k).startswith("val_") else v) for k, v in rec.items()}
            for rec in grp.to_dict("records")
        ]
        best_masked = max(masked_rows, key=cv_selection_key_for_svm)
        assert (
            str(sel_row["kernel"]),
            str(sel_row["class_weight"]),
            float(sel_row["C"]),
            str(sel_row["gamma"]),
        ) == (
            str(best_masked["kernel"]),
            str(best_masked["class_weight"]),
            float(best_masked["C"]),
            str(best_masked["gamma"]),
        )

    # Verify all 15 exported cv_selected JSON models match svm_2x3_val_predictions.csv & summary
    _, val_app, _ = load_shared_appearance_frame(REPO_ROOT)
    _, val_seg, _ = load_classification_feature_frame(REPO_ROOT, method="k4_M0", segmented=True)
    val_frames = {m: load_classification_feature_frame(REPO_ROOT, method=m, segmented=False)[1] for m in COMPARISON_2X3_METHODS}

    assert len(manifest["exported_model_files"]) == 15
    summary_cv = summary_df.loc[summary_df["model_variant"].eq("cv_selected")].set_index("model_id")

    for rel_path in manifest["exported_model_files"]:
        model_path = REPO_ROOT / rel_path
        assert model_path.parent == REPO_ROOT / MODELS_CV_2X3_DIR_REL
        artifact = load_svm_artifact(model_path)
        assert int(artifact["scaler"]["n_samples_seen"]) == 1535

        method = str(artifact["segmentation_method"])
        set_name = str(artifact["feature_set"])
        cols = feature_columns_for_set(set_name)
        if method == SHARED_APPEARANCE_METHOD:
            va_df = val_app
        elif set_name in ("A_seg", "C_seg"):
            va_df = val_seg
        else:
            va_df = val_frames[method]

        preds, _, _ = predict_from_svm_artifact(va_df[list(cols)].to_numpy(dtype=np.float64), artifact)
        stored_preds = (
            pred_df.loc[pred_df["model_id"].eq(artifact["model_id"])]
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        assert np.array_equal(preds, stored_preds["predicted_category"].to_numpy(dtype=object))
        metrics = classification_metrics(va_df["category"], preds)
        assert abs(metrics["macro_f1"] - float(summary_cv.loc[artifact["model_id"], "val_macro_f1"])) < 1e-12


def test_fit_fold_otsu_polarity_uses_only_fold_train() -> None:
    fold_train_df = pd.DataFrame(
        {
            "sample_id": ["tr_a1", "tr_a2", "tr_b1", "tr_b2"],
            "category": ["Adenocarcinoma", "Adenocarcinoma", "Polyp", "Polyp"],
            "group_id": ["G1", "G1", "G2", "G2"],
        }
    )
    precomputed_dice = pd.DataFrame(
        {
            "sample_id": ["tr_a1", "tr_a2", "tr_b1", "tr_b2", "va_holdout_1", "va_holdout_2"],
            "category": ["Adenocarcinoma", "Adenocarcinoma", "Polyp", "Polyp", "Polyp", "Adenocarcinoma"],
            "dark_dice": [0.20, 0.25, 0.30, 0.35, 1.00, 1.00],
            "dark_iou": [0.11, 0.14, 0.18, 0.21, 1.00, 1.00],
            "bright_dice": [0.85, 0.90, 0.80, 0.88, 0.00, 0.00],
            "bright_iou": [0.74, 0.82, 0.67, 0.79, 0.00, 0.00],
        }
    )
    fit = fit_fold_otsu_polarity(fold_train_df, REPO_ROOT, precomputed_dice_df=precomputed_dice)
    assert fit["chosen_polarity"] == "bright"
    assert fit["n_train_images"] == 4
    assert fit["n_train_groups"] == 2
    assert float(fit["bright_mean_category_dice"]) > float(fit["dark_mean_category_dice"])


def test_fit_fold_kmeans_k4_mapped_isolated_from_holdout() -> None:
    fold_train_df = pd.DataFrame(
        {
            "sample_id": [f"tr_{i}" for i in range(4)],
            "image_path": [f"tr_{i}.png" for i in range(4)],
            "mask_path": [f"tr_{i}_m.png" for i in range(4)],
            "group_id": ["G1", "G1", "G2", "G2"],
        }
    )
    rng = np.random.default_rng(7)
    base_colors = [
        np.array([40, 20, 40], dtype=np.uint8),
        np.array([120, 40, 120], dtype=np.uint8),
        np.array([180, 120, 180], dtype=np.uint8),
        np.array([235, 220, 235], dtype=np.uint8),
    ]
    assets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(4):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        ref = np.zeros((32, 32), dtype=bool)
        for idx, col in enumerate(base_colors):
            r0, r1 = idx * 8, (idx + 1) * 8
            noise = rng.integers(-4, 5, size=(8, 32, 3), dtype=np.int16)
            img[r0:r1] = np.clip(col.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            if idx < 2:
                ref[r0:r1] = True
        assets[f"tr_{i}"] = (img, ref)

    assets["va_holdout"] = (np.full((32, 32, 3), 10, dtype=np.uint8), np.ones((32, 32), dtype=bool))
    fit_a = fit_fold_kmeans_k4_mapped(
        fold_train_df, REPO_ROOT, k=4, pixels_per_image=256, seed=42, _preloaded_assets=assets
    )
    assets["va_holdout"] = (np.full((32, 32, 3), 250, dtype=np.uint8), np.zeros((32, 32), dtype=bool))
    fit_b = fit_fold_kmeans_k4_mapped(
        fold_train_df, REPO_ROOT, k=4, pixels_per_image=256, seed=42, _preloaded_assets=assets
    )
    assert fit_a["training_images"] == 4
    assert fit_a["train_sample_ids_sha256"] == fit_b["train_sample_ids_sha256"]
    assert fit_a["foreground_clusters"] == fit_b["foreground_clusters"]
    np.testing.assert_allclose(fit_a["cluster_centers"], fit_b["cluster_centers"], rtol=0.0, atol=0.0)


def test_evaluate_strict_cv_grid_is_validation_blind_and_deterministic() -> None:
    rng = np.random.default_rng(123)
    y_parts = []
    X_parts = []
    for cls_idx, cat in enumerate(EXPECTED_CATEGORIES):
        y_parts.extend([cat] * 6)
        X_parts.append(rng.normal(loc=cls_idx * 2.5, scale=0.8, size=(6, 3)))
    y_train = np.asarray(y_parts, dtype=object)
    X_base = np.vstack(X_parts)
    fold_splits = [
        (np.asarray([i for i in range(36) if i % 3 != f]), np.asarray([i for i in range(36) if i % 3 == f]))
        for f in range(3)
    ]
    fold_mats = [X_base + 0.01 * f for f in range(3)]
    grid_1, best_1, _ = evaluate_strict_cv_grid(
        y_train=y_train,
        fold_splits=fold_splits,
        fold_feature_matrices=fold_mats,
        feature_names=("f0", "f1", "f2"),
        method="otsu_M0",
        feature_set="B",
        track="factorial_2x3",
    )
    grid_2, best_2, _ = evaluate_strict_cv_grid(
        y_train=y_train,
        fold_splits=fold_splits,
        fold_feature_matrices=fold_mats,
        feature_names=("f0", "f1", "f2"),
        method="otsu_M0",
        feature_set="B",
        track="factorial_2x3",
    )
    assert len(grid_1) == 32
    assert not any(str(col).startswith("val_") for col in grid_1.columns)
    assert int(grid_1["is_selected_best"].sum()) == 1
    assert int(grid_1["is_default_baseline"].sum()) == 1
    assert best_1 == best_2
    pd.testing.assert_frame_equal(grid_1, grid_2)


def test_saved_stage07_2x3_strict_artifacts_bootstrap_and_parity() -> None:
    strict_dir = REPO_ROOT / CLASSIFICATION_2X3_STRICT_DIR_REL
    manifest = json.loads((strict_dir / "classification_2x3_strict_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((strict_dir / "fold_upstream_audit.json").read_text(encoding="utf-8"))

    assert manifest["protocol_id"] == "stage07_2x3_strict_fold_aware_val_blind_cv_v3"
    assert manifest["cohort_counts"]["test_images_touched_in_2x3_strict"] == 0
    assert manifest["cohort_counts"]["test_labels_accessed_in_2x3_strict"] is False
    assert manifest["preserved_v1_classification_manifest_sha256"] == sha256_bytes(
        REPO_ROOT / "results/classification/classification_manifest.json"
    )
    assert manifest["preserved_v2_classification_manifest_sha256"] == sha256_bytes(
        REPO_ROOT / CLASSIFICATION_2X3_DIR_REL / "classification_2x3_manifest.json"
    )

    assert len(audit["folds"]) == 5
    assert audit["otsu_fold_invariance_all_dark"] is True
    assert all(bool(f["group_id_disjoint"]) and bool(f["sample_id_disjoint"]) for f in audit["folds"])

    grid_df = pd.read_csv(strict_dir / "svm_2x3_strict_grid_results.csv")
    summary_df = pd.read_csv(strict_dir / "svm_2x3_strict_selected_summary.csv")
    pred_df = pd.read_csv(strict_dir / "svm_2x3_strict_val_predictions.csv")
    boot_models_df = pd.read_csv(strict_dir / "svm_2x3_strict_val_bootstrap_models.csv")

    assert len(grid_df) == 15 * 32
    assert not any(str(col).startswith("val_") for col in grid_df.columns)
    assert len(summary_df) == 15 * 2
    assert set(pred_df["split"]) == {"val"}

    re_boot_models, _ = compute_val_bootstrap_models_and_contrasts(pred_df, n_resamples=2000, seed=42)
    np.testing.assert_allclose(
        re_boot_models["val_macro_f1_ci95_low"].to_numpy(),
        boot_models_df["val_macro_f1_ci95_low"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        re_boot_models["val_macro_f1_ci95_high"].to_numpy(),
        boot_models_df["val_macro_f1_ci95_high"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )

    _, val_app, _ = load_shared_appearance_frame(REPO_ROOT)
    _, val_seg, _ = load_classification_feature_frame(REPO_ROOT, method="k4_M0", segmented=True)
    val_frames = {m: load_classification_feature_frame(REPO_ROOT, method=m, segmented=False)[1] for m in COMPARISON_2X3_METHODS}
    summary_cv = summary_df.loc[summary_df["model_variant"].eq("cv_selected")].set_index("model_id")

    assert len(manifest["exported_model_files"]) == 15
    for rel_path in manifest["exported_model_files"]:
        model_path = REPO_ROOT / rel_path
        assert model_path.parent == REPO_ROOT / MODELS_CV_2X3_STRICT_DIR_REL
        artifact = load_svm_artifact(model_path)
        assert artifact.get("cv_protocol") == "5fold_stratified_group_kfold_strict_fold_aware_upstream"
        method = str(artifact["segmentation_method"])
        set_name = str(artifact["feature_set"])
        cols = feature_columns_for_set(set_name)
        va_df = (
            val_app
            if method == SHARED_APPEARANCE_METHOD
            else (val_seg if set_name in ("A_seg", "C_seg") else val_frames[method])
        )
        preds, _, _ = predict_from_svm_artifact(va_df[list(cols)].to_numpy(dtype=np.float64), artifact)
        stored_preds = (
            pred_df.loc[pred_df["model_id"].eq(artifact["model_id"]) & pred_df["model_variant"].eq("cv_selected")]
            .sort_values("sample_id", kind="mergesort")
            .reset_index(drop=True)
        )
        assert np.array_equal(preds, stored_preds["predicted_category"].to_numpy(dtype=object))
        metrics = classification_metrics(va_df["category"], preds)
        assert abs(metrics["macro_f1"] - float(summary_cv.loc[artifact["model_id"], "val_macro_f1"])) < 1e-12


