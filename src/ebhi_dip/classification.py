"""Six-class image-level SVM classification, group-aware CV, and portable JSON model export.

Stage 07 contract:
- Uses only ``train`` (1,535 images, 247 groups) for scaler fitting, 5-fold StratifiedGroupKFold
  cross-validation, and final training-split model fitting.
- Uses ``val`` (372 images, 53 groups) for feature-set comparison (A, B, C), hyperparameter
  selection, A_seg/C_seg sensitivity, and segmentation comparator sensitivity
  (otsu_M0, otsu_M2, k4_M0, k4_M1).
- Never accesses held-out ``test`` (319 images) for scaling, tuning, model selection, or evaluation.
- Exports self-contained JSON scaler + SVM artifacts verified against scikit-learn to ``<= 1e-10``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ebhi_dip.dataset import EXPECTED_CATEGORIES, load_split_manifest
from ebhi_dip.features import (
    APPEARANCE_FEATURE_NAMES,
    DEFAULT_FEATURE_METHODS,
    FEATURE_NAMES,
    METADATA_COLUMNS,
    MORPHOLOGY_FEATURE_NAMES,
    PRIMARY_FEATURE_METHOD,
    SEGMENTED_APPEARANCE_FEATURE_NAMES,
    SEGMENTED_FEATURE_NAMES,
    validate_feature_table,
)
from ebhi_dip.paths import verify_frozen_split
from ebhi_dip.provenance import sha256_bytes

PRIMARY_FEATURE_SETS: tuple[str, ...] = ("A", "B", "C")
SENSITIVITY_FEATURE_SETS: tuple[str, ...] = ("A_seg", "C_seg")
ALL_FEATURE_SETS: tuple[str, ...] = PRIMARY_FEATURE_SETS + SENSITIVITY_FEATURE_SETS

SVM_KERNELS: tuple[str, ...] = ("linear", "rbf")
SVM_CLASS_WEIGHTS: tuple[str, ...] = ("none", "balanced")
SVM_C_VALUES: tuple[float, ...] = (0.1, 1.0, 3.0, 10.0)
SVM_RBF_GAMMAS: tuple[str | float, ...] = ("scale", 0.01, 0.1)
CV_FOLDS: int = 5
DECISION_PARITY_ATOL: float = 1e-10

DEFAULT_BASELINE_SPEC: dict[str, object] = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "none",
}


@dataclass(frozen=True)
class SvmExperimentOutputs:
    """Container for Stage 07 validation tables, predictions, and portable model artifacts."""

    grid_results: pd.DataFrame
    selected_summary: pd.DataFrame
    per_category_val: pd.DataFrame
    comparator_sensitivity: pd.DataFrame
    val_predictions: pd.DataFrame
    model_artifacts: dict[str, dict[str, object]]
    manifest: dict[str, object]
    split_sha256: str


def feature_columns_for_set(feature_set: str) -> tuple[str, ...]:
    """Return the ordered feature names for set A, B, C, A_seg, or C_seg."""
    mapping: dict[str, tuple[str, ...]] = {
        "A": APPEARANCE_FEATURE_NAMES,
        "B": MORPHOLOGY_FEATURE_NAMES,
        "C": FEATURE_NAMES,
        "A_seg": SEGMENTED_APPEARANCE_FEATURE_NAMES,
        "C_seg": SEGMENTED_FEATURE_NAMES,
    }
    if feature_set not in mapping:
        raise ValueError(
            f"Unknown feature_set {feature_set!r}. Expected one of {sorted(mapping)}."
        )
    return mapping[feature_set]


def _format_gamma(gamma: str | float) -> str:
    if isinstance(gamma, str):
        return gamma
    return f"{float(gamma):g}"


def _parse_gamma(gamma: str | float) -> str | float:
    if isinstance(gamma, (int, float)):
        return float(gamma)
    text = str(gamma).strip()
    if text in ("scale", "auto"):
        return text
    return float(text)


def load_classification_feature_frame(
    project_root: str | Path,
    method: str = PRIMARY_FEATURE_METHOD,
    segmented: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load and validate strictly ``train`` and ``val`` feature rows (excluding ``test``)."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    split_df = load_split_manifest(root, verify=False)

    feature_dir = root / "results/features"
    if segmented:
        csv_path = feature_dir / f"features_{method}_segmented.csv"
        expected_cols = SEGMENTED_FEATURE_NAMES
    else:
        csv_path = feature_dir / f"features_{method}.csv"
        expected_cols = FEATURE_NAMES

    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing canonical feature table: {csv_path}")

    table = pd.read_csv(csv_path)
    validate_feature_table(table, feature_names=expected_cols, require_method=True)

    if not table["method"].eq(method).all():
        raise ValueError(f"Feature table {csv_path.name} contains method values other than {method!r}.")

    # Verify alignment with frozen split_manifest before filtering out held-out test rows
    merged_check = table[list(METADATA_COLUMNS)].merge(
        split_df[list(METADATA_COLUMNS)],
        on="sample_id",
        suffixes=("_feat", "_split"),
        how="outer",
        validate="one_to_one",
    )
    for col in ("category", "group_id", "split"):
        if not merged_check[f"{col}_feat"].eq(merged_check[f"{col}_split"]).all():
            raise ValueError(f"Metadata mismatch on column '{col}' in {csv_path.name}.")

    train_df = (
        table.loc[table["split"].eq("train")]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    val_df = (
        table.loc[table["split"].eq("val")]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )

    if len(train_df) != 1535 or len(val_df) != 372:
        raise ValueError(
            f"Unexpected train/val row counts: train={len(train_df)}, val={len(val_df)}."
        )
    if not set(train_df["sample_id"]).isdisjoint(set(val_df["sample_id"])):
        raise ValueError("Train and val sample_id sets must be disjoint.")
    if not set(train_df["group_id"]).isdisjoint(set(val_df["group_id"])):
        raise ValueError("Train and val group_id sets must be disjoint.")
    if "test" in set(train_df["split"]) or "test" in set(val_df["split"]):
        raise RuntimeError("Held-out test split leaked into train/val frames.")

    return train_df, val_df, split_sha


def fit_standard_scaler(
    train_matrix: np.ndarray,
    feature_names: Sequence[str],
) -> dict[str, object]:
    """Fit a population-std StandardScaler on training features and return a JSON-ready dict."""
    arr = np.asarray(train_matrix, dtype=np.float64)
    names = tuple(str(name) for name in feature_names)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(
            f"Expected 2D matrix with {len(names)} columns, got shape {arr.shape}."
        )
    if arr.shape[0] < 2:
        raise ValueError("At least 2 training samples are required to fit StandardScaler.")
    if not np.isfinite(arr).all():
        raise ValueError("Training matrix contains NaN or infinite values.")

    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    scale = np.where(std > 0.0, std, 1.0)
    return {
        "feature_names": list(names),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "scale": [float(v) for v in scale],
        "n_samples_seen": int(arr.shape[0]),
    }


def apply_standard_scaler(
    matrix: np.ndarray,
    scaler_payload: Mapping[str, object],
) -> np.ndarray:
    """Standardize a feature matrix using a saved scaler payload."""
    arr = np.asarray(matrix, dtype=np.float64)
    mean = np.asarray(scaler_payload["mean"], dtype=np.float64)
    scale = np.asarray(scaler_payload["scale"], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Matrix shape {arr.shape} incompatible with scaler of length {mean.shape[0]}."
        )
    if np.any(scale <= 0.0):
        raise ValueError("Scaler scale values must be strictly positive.")
    return (arr - mean) / scale


def classification_metrics(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    expected_categories: Sequence[str] = EXPECTED_CATEGORIES,
) -> dict[str, float]:
    """Compute six-class macro F1, balanced accuracy, accuracy, weighted F1, and macro precision."""
    labels = list(expected_categories)
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    if yt.shape != yp.shape or yt.ndim != 1 or yt.size == 0:
        raise ValueError("y_true and y_pred must be non-empty 1D arrays of equal length.")

    unknown = sorted((set(yt) | set(yp)) - set(labels))
    if unknown:
        raise ValueError(f"Unexpected category labels encountered: {unknown}.")

    return {
        "macro_f1": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "accuracy": float(accuracy_score(yt, yp)),
        "weighted_f1": float(f1_score(yt, yp, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(yt, yp, labels=labels, average="macro", zero_division=0)
        ),
    }


def per_category_classification_table(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    expected_categories: Sequence[str] = EXPECTED_CATEGORIES,
) -> pd.DataFrame:
    """Compute per-category support, TP, FP, FN, TN, precision, recall, and F1."""
    labels = list(expected_categories)
    yt = np.asarray(y_true, dtype=object)
    yp = np.asarray(y_pred, dtype=object)
    cm = confusion_matrix(yt, yp, labels=labels)
    total = int(cm.sum())

    prec = precision_score(yt, yp, labels=labels, average=None, zero_division=0)
    rec = recall_score(yt, yp, labels=labels, average=None, zero_division=0)
    f1 = f1_score(yt, yp, labels=labels, average=None, zero_division=0)

    rows = []
    for idx, cat in enumerate(labels):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(total - tp - fn - fp)
        rows.append(
            {
                "category": cat,
                "support": int(cm[idx, :].sum()),
                "predicted_count": int(cm[:, idx].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(prec[idx]),
                "recall": float(rec[idx]),
                "f1": float(f1[idx]),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_table(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    expected_categories: Sequence[str] = EXPECTED_CATEGORIES,
    normalize: str | None = None,
) -> pd.DataFrame:
    """Return a 6x6 confusion matrix DataFrame (raw counts or row-normalized recall)."""
    labels = list(expected_categories)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    dtype = float if normalize is not None else int
    return pd.DataFrame(cm.astype(dtype), index=labels, columns=labels)


def selection_key_for_svm(row: Mapping[str, object]) -> tuple[float, float, float, float, int, float]:
    """Deterministic model selection rule for Stage 07 validation candidates.

    Order of precedence:
    1. Validation Macro F1 rounded to 4 decimals (equal-category primary metric)
    2. Validation Balanced Accuracy rounded to 4 decimals (equal-category recall)
    3. Validation Accuracy rounded to 4 decimals
    4. 5-fold train Group-CV Macro F1 rounded to 4 decimals (stability tie-breaker)
    5. Simpler kernel (linear=1 > rbf=0)
    6. Stronger regularization (-C)
    """
    kernel_simplicity = 1 if str(row["kernel"]) == "linear" else 0
    return (
        round(float(row["val_macro_f1"]), 4),
        round(float(row["val_balanced_accuracy"]), 4),
        round(float(row["val_accuracy"]), 4),
        round(float(row.get("cv_macro_f1", 0.0)), 4),
        kernel_simplicity,
        -float(row["C"]),
    )


def fit_svm_model(
    X_train_scaled: np.ndarray,
    y_train: Sequence[str] | np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight: str = "none",
) -> SVC:
    """Fit a deterministic multi-class one-vs-one SVC classifier on scaled training features."""
    if kernel not in SVM_KERNELS:
        raise ValueError(f"Unsupported SVM kernel: {kernel!r}.")
    if class_weight not in SVM_CLASS_WEIGHTS:
        raise ValueError(f"Unsupported class_weight: {class_weight!r}.")
    cw_arg = None if class_weight == "none" else "balanced"
    gamma_arg = "scale" if kernel == "linear" else _parse_gamma(gamma)

    clf = SVC(
        kernel=kernel,
        C=float(C),
        gamma=gamma_arg,
        class_weight=cw_arg,
        decision_function_shape="ovo",
        tol=1e-3,
        cache_size=200,
    )
    clf.fit(np.asarray(X_train_scaled, dtype=np.float64), np.asarray(y_train, dtype=object))
    return clf


def export_svm_artifact(
    model_id: str,
    segmentation_method: str,
    feature_set: str,
    scaler_payload: Mapping[str, object],
    clf: SVC,
    split_sha256: str,
    validation_metrics: Mapping[str, float],
    cv_metrics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Export a fitted StandardScaler + SVC into a portable JSON-serializable dictionary."""
    classes = [str(c) for c in clf.classes_]
    n_classes = len(classes)
    n_support = [int(v) for v in clf.n_support_]
    sv = np.asarray(clf.support_vectors_, dtype=np.float64)
    dual_coef = np.asarray(clf.dual_coef_, dtype=np.float64)
    intercept = np.asarray(clf.intercept_, dtype=np.float64)
    gamma_val = float(clf._gamma)

    ovo_pairs: list[list[int]] = []
    linear_weights: list[list[float]] = []
    starts = np.cumsum(np.r_[0, n_support])
    for i in range(n_classes):
        si, ei = int(starts[i]), int(starts[i + 1])
        for j in range(i + 1, n_classes):
            sj, ej = int(starts[j]), int(starts[j + 1])
            ovo_pairs.append([i, j])
            if clf.kernel == "linear":
                w_ij = dual_coef[j - 1, si:ei] @ sv[si:ei] + dual_coef[i, sj:ej] @ sv[sj:ej]
                linear_weights.append([float(v) for v in w_ij])

    cw_label = "balanced" if clf.class_weight == "balanced" else "none"
    gamma_label = "scale" if clf.kernel == "linear" else _format_gamma(clf.gamma)

    artifact: dict[str, object] = {
        "schema_version": "svm-classifier-v1",
        "model_id": model_id,
        "segmentation_method": segmentation_method,
        "feature_set": feature_set,
        "feature_names": list(scaler_payload["feature_names"]),
        "n_features": len(scaler_payload["feature_names"]),
        "classes": classes,
        "kernel": str(clf.kernel),
        "C": float(clf.C),
        "gamma_param": gamma_label,
        "gamma_value": gamma_val,
        "class_weight": cw_label,
        "decision_scheme": "libsvm-one-vs-one-majority-vote-lowest-index-tiebreak",
        "split_sha256": split_sha256,
        "scaler": dict(scaler_payload),
        "ovo_pairs": ovo_pairs,
        "n_support": n_support,
        "support_vectors": [[float(x) for x in row] for row in sv],
        "dual_coef": [[float(x) for x in row] for row in dual_coef],
        "intercept": [float(x) for x in intercept],
        "validation_metrics": {k: float(v) for k, v in validation_metrics.items()},
    }
    if cv_metrics is not None:
        artifact["train_cv_metrics"] = {k: float(v) for k, v in cv_metrics.items()}
    if clf.kernel == "linear":
        artifact["linear_ovo_weights"] = linear_weights
    return artifact


def predict_from_svm_artifact(
    raw_matrix: np.ndarray,
    artifact: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a portable SVM JSON artifact in pure NumPy (C++ reference implementation).

    Returns:
        predicted_labels: 1D object array of predicted category strings
        decision_ovo: 2D float64 array of shape (N, 15) one-vs-one decision values
        votes: 2D int32 array of shape (N, 6) one-vs-one class vote counts
    """
    X_scaled = apply_standard_scaler(raw_matrix, artifact["scaler"])
    classes = [str(c) for c in artifact["classes"]]
    n_classes = len(classes)
    n_samples = X_scaled.shape[0]
    n_pairs = n_classes * (n_classes - 1) // 2

    intercept = np.asarray(artifact["intercept"], dtype=np.float64)
    kernel = str(artifact["kernel"])

    if kernel == "linear" and "linear_ovo_weights" in artifact:
        W = np.asarray(artifact["linear_ovo_weights"], dtype=np.float64)
        dec_ovo = X_scaled @ W.T + intercept[None, :]
    else:
        sv = np.asarray(artifact["support_vectors"], dtype=np.float64)
        dual_coef = np.asarray(artifact["dual_coef"], dtype=np.float64)
        n_support = np.asarray(artifact["n_support"], dtype=int)
        gamma_val = float(artifact["gamma_value"])

        if kernel == "linear":
            K_mat = X_scaled @ sv.T
        elif kernel == "rbf":
            sq_dist = np.sum((X_scaled[:, None, :] - sv[None, :, :]) ** 2, axis=2)
            K_mat = np.exp(-gamma_val * sq_dist)
        else:
            raise ValueError(f"Unsupported kernel in artifact: {kernel!r}.")

        starts = np.cumsum(np.r_[0, n_support])
        dec_ovo = np.zeros((n_samples, n_pairs), dtype=np.float64)
        pair_idx = 0
        for i in range(n_classes):
            si, ei = int(starts[i]), int(starts[i + 1])
            for j in range(i + 1, n_classes):
                sj, ej = int(starts[j]), int(starts[j + 1])
                dec_ovo[:, pair_idx] = (
                    K_mat[:, si:ei] @ dual_coef[j - 1, si:ei]
                    + K_mat[:, sj:ej] @ dual_coef[i, sj:ej]
                    + intercept[pair_idx]
                )
                pair_idx += 1

    votes = np.zeros((n_samples, n_classes), dtype=np.int32)
    pair_idx = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            win_i = dec_ovo[:, pair_idx] > 0.0
            votes[win_i, i] += 1
            votes[~win_i, j] += 1
            pair_idx += 1

    pred_indices = np.argmax(votes, axis=1)
    pred_labels = np.asarray([classes[int(idx)] for idx in pred_indices], dtype=object)
    return pred_labels, dec_ovo, votes


def verify_svm_artifact_parity(
    raw_matrix: np.ndarray,
    clf: SVC,
    artifact: Mapping[str, object],
    atol: float = DECISION_PARITY_ATOL,
) -> dict[str, float | bool]:
    """Verify that pure-NumPy artifact inference matches sklearn SVC on a feature matrix."""
    X_scaled = apply_standard_scaler(raw_matrix, artifact["scaler"])
    sk_dec = np.asarray(clf.decision_function(X_scaled), dtype=np.float64)
    sk_pred = np.asarray(clf.predict(X_scaled), dtype=object)

    art_pred, art_dec, _ = predict_from_svm_artifact(raw_matrix, artifact)
    max_abs_diff = float(np.max(np.abs(art_dec - sk_dec)))
    exact_match = bool(np.array_equal(art_pred, sk_pred))
    if max_abs_diff > atol or not exact_match:
        raise ValueError(
            f"SVM artifact parity check failed: max_abs_diff={max_abs_diff:.3e} "
            f"(atol={atol:.1e}), exact_match={exact_match}."
        )
    return {
        "max_abs_decision_diff": max_abs_diff,
        "exact_prediction_match": exact_match,
    }


def _evaluate_grid_for_table(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    method: str,
    feature_set: str,
    feature_names: Sequence[str],
    n_splits: int = CV_FOLDS,
) -> tuple[pd.DataFrame, dict[str, object], SVC, dict[str, object], dict[str, object]]:
    """Run 5-fold StratifiedGroupKFold on train + validation evaluation for one feature set."""
    cols = list(feature_names)
    X_tr_raw = train_df[cols].to_numpy(dtype=np.float64)
    y_tr = train_df["category"].to_numpy(dtype=object)
    g_tr = train_df["group_id"].to_numpy(dtype=object)
    X_va_raw = val_df[cols].to_numpy(dtype=np.float64)
    y_va = val_df["category"].to_numpy(dtype=object)

    scaler_payload = fit_standard_scaler(X_tr_raw, cols)
    X_tr = apply_standard_scaler(X_tr_raw, scaler_payload)
    X_va = apply_standard_scaler(X_va_raw, scaler_payload)

    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    fold_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for f_tr_idx, f_va_idx in sgkf.split(X_tr_raw, y_tr, g_tr):
        sc_fold = fit_standard_scaler(X_tr_raw[f_tr_idx], cols)
        fold_cache.append(
            (
                f_va_idx,
                apply_standard_scaler(X_tr_raw[f_tr_idx], sc_fold),
                y_tr[f_tr_idx],
                apply_standard_scaler(X_tr_raw[f_va_idx], sc_fold),
            )
        )

    rows: list[dict[str, object]] = []
    fitted_models: dict[tuple[str, str, float, str], SVC] = {}

    for kernel in SVM_KERNELS:
        gammas: Sequence[str | float] = ("scale",) if kernel == "linear" else SVM_RBF_GAMMAS
        for cw in SVM_CLASS_WEIGHTS:
            for C in SVM_C_VALUES:
                for gamma in gammas:
                    gamma_str = _format_gamma(gamma)
                    oof_pred = np.empty_like(y_tr)
                    for f_va_idx, Xf_tr, yf_tr, Xf_va in fold_cache:
                        clf_fold = fit_svm_model(
                            Xf_tr, yf_tr, kernel=kernel, C=C, gamma=gamma, class_weight=cw
                        )
                        oof_pred[f_va_idx] = clf_fold.predict(Xf_va)
                    cv_scores = classification_metrics(y_tr, oof_pred)

                    clf = fit_svm_model(
                        X_tr, y_tr, kernel=kernel, C=C, gamma=gamma, class_weight=cw
                    )
                    tr_pred = clf.predict(X_tr)
                    va_pred = clf.predict(X_va)
                    tr_scores = classification_metrics(y_tr, tr_pred)
                    va_scores = classification_metrics(y_va, va_pred)

                    key = (kernel, cw, float(C), gamma_str)
                    fitted_models[key] = clf
                    rows.append(
                        {
                            "method": method,
                            "feature_set": feature_set,
                            "n_features": len(cols),
                            "kernel": kernel,
                            "class_weight": cw,
                            "C": float(C),
                            "gamma": gamma_str,
                            "train_macro_f1": tr_scores["macro_f1"],
                            "train_balanced_accuracy": tr_scores["balanced_accuracy"],
                            "train_accuracy": tr_scores["accuracy"],
                            "cv_macro_f1": cv_scores["macro_f1"],
                            "cv_balanced_accuracy": cv_scores["balanced_accuracy"],
                            "cv_accuracy": cv_scores["accuracy"],
                            "cv_weighted_f1": cv_scores["weighted_f1"],
                            "val_macro_f1": va_scores["macro_f1"],
                            "val_balanced_accuracy": va_scores["balanced_accuracy"],
                            "val_accuracy": va_scores["accuracy"],
                            "val_weighted_f1": va_scores["weighted_f1"],
                            "val_macro_precision": va_scores["macro_precision"],
                            "n_support_total": int(np.sum(clf.n_support_)),
                        }
                    )

    best_row = max(rows, key=selection_key_for_svm)
    default_row = next(
        r
        for r in rows
        if r["kernel"] == DEFAULT_BASELINE_SPEC["kernel"]
        and float(r["C"]) == float(DEFAULT_BASELINE_SPEC["C"])
        and r["gamma"] == str(DEFAULT_BASELINE_SPEC["gamma"])
        and r["class_weight"] == str(DEFAULT_BASELINE_SPEC["class_weight"])
    )
    best_key = (
        str(best_row["kernel"]),
        str(best_row["class_weight"]),
        float(best_row["C"]),
        str(best_row["gamma"]),
    )
    best_clf = fitted_models[best_key]

    grid_df = pd.DataFrame(rows)
    grid_df["is_default_baseline"] = (
        grid_df["kernel"].eq(default_row["kernel"])
        & grid_df["C"].eq(default_row["C"])
        & grid_df["gamma"].eq(default_row["gamma"])
        & grid_df["class_weight"].eq(default_row["class_weight"])
    )
    grid_df["is_selected_best"] = (
        grid_df["kernel"].eq(best_row["kernel"])
        & grid_df["C"].eq(best_row["C"])
        & grid_df["gamma"].eq(best_row["gamma"])
        & grid_df["class_weight"].eq(best_row["class_weight"])
    )
    return grid_df, scaler_payload, best_clf, dict(best_row), dict(default_row)


def _prediction_rows_for_model(
    val_df: pd.DataFrame,
    raw_val_matrix: np.ndarray,
    artifact: Mapping[str, object],
    model_variant: str,
) -> pd.DataFrame:
    pred_labels, _, votes = predict_from_svm_artifact(raw_val_matrix, artifact)
    sorted_votes = np.sort(votes, axis=1)
    vote_margin = sorted_votes[:, -1] - sorted_votes[:, -2]
    max_votes = sorted_votes[:, -1]

    out = val_df[list(METADATA_COLUMNS)].copy()
    out["method"] = str(artifact["segmentation_method"])
    out["feature_set"] = str(artifact["feature_set"])
    out["model_variant"] = model_variant
    out["predicted_category"] = pred_labels
    out["correct"] = out["category"].to_numpy() == pred_labels
    out["winning_votes"] = max_votes.astype(int)
    out["vote_margin"] = vote_margin.astype(int)
    return out


def run_svm_experiments(
    project_root: str | Path,
    primary_method: str = PRIMARY_FEATURE_METHOD,
    comparators: Sequence[str] = DEFAULT_FEATURE_METHODS,
) -> SvmExperimentOutputs:
    """Execute the complete Stage 07 train/val SVM study without touching held-out test."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)

    train_primary, val_primary, _ = load_classification_feature_frame(
        root, method=primary_method, segmented=False
    )
    train_seg, val_seg, _ = load_classification_feature_frame(
        root, method=primary_method, segmented=True
    )

    grid_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    per_cat_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    model_artifacts: dict[str, dict[str, object]] = {}
    parity_checks: dict[str, dict[str, float | bool]] = {}

    set_sources = {
        "A": (train_primary, val_primary),
        "B": (train_primary, val_primary),
        "C": (train_primary, val_primary),
        "A_seg": (train_seg, val_seg),
        "C_seg": (train_seg, val_seg),
    }

    for set_name in ALL_FEATURE_SETS:
        tr_df, va_df = set_sources[set_name]
        cols = feature_columns_for_set(set_name)
        grid_df, scaler_payload, best_clf, best_row, default_row = _evaluate_grid_for_table(
            tr_df, va_df, method=primary_method, feature_set=set_name, feature_names=cols
        )
        grid_frames.append(grid_df)

        # Also fit default baseline model for side-by-side comparison and predictions
        X_tr_scaled = apply_standard_scaler(tr_df[list(cols)].to_numpy(dtype=np.float64), scaler_payload)
        y_tr = tr_df["category"].to_numpy(dtype=object)
        X_va_raw = va_df[list(cols)].to_numpy(dtype=np.float64)
        y_va = va_df["category"].to_numpy(dtype=object)

        default_clf = fit_svm_model(
            X_tr_scaled,
            y_tr,
            kernel=str(default_row["kernel"]),
            C=float(default_row["C"]),
            gamma=str(default_row["gamma"]),
            class_weight=str(default_row["class_weight"]),
        )

        for variant_name, row_info, clf_obj in (
            ("default_rbf", default_row, default_clf),
            ("selected_best", best_row, best_clf),
        ):
            summary_rows.append(
                {
                    "track": "primary_k4_M0" if set_name in PRIMARY_FEATURE_SETS else "sensitivity_A_seg_C_seg",
                    "method": primary_method,
                    "feature_set": set_name,
                    "n_features": len(cols),
                    "model_variant": variant_name,
                    "kernel": row_info["kernel"],
                    "class_weight": row_info["class_weight"],
                    "C": float(row_info["C"]),
                    "gamma": row_info["gamma"],
                    "train_macro_f1": float(row_info["train_macro_f1"]),
                    "train_accuracy": float(row_info["train_accuracy"]),
                    "cv_macro_f1": float(row_info["cv_macro_f1"]),
                    "cv_balanced_accuracy": float(row_info["cv_balanced_accuracy"]),
                    "cv_accuracy": float(row_info["cv_accuracy"]),
                    "val_macro_f1": float(row_info["val_macro_f1"]),
                    "val_balanced_accuracy": float(row_info["val_balanced_accuracy"]),
                    "val_accuracy": float(row_info["val_accuracy"]),
                    "val_weighted_f1": float(row_info["val_weighted_f1"]),
                    "val_macro_precision": float(row_info["val_macro_precision"]),
                    "n_support_total": int(row_info["n_support_total"]),
                }
            )

            art_id = f"svm_{primary_method}_{set_name}_{variant_name}"
            art = export_svm_artifact(
                model_id=art_id,
                segmentation_method=primary_method,
                feature_set=set_name,
                scaler_payload=scaler_payload,
                clf=clf_obj,
                split_sha256=split_sha,
                validation_metrics={
                    "macro_f1": float(row_info["val_macro_f1"]),
                    "balanced_accuracy": float(row_info["val_balanced_accuracy"]),
                    "accuracy": float(row_info["val_accuracy"]),
                    "weighted_f1": float(row_info["val_weighted_f1"]),
                    "macro_precision": float(row_info["val_macro_precision"]),
                },
                cv_metrics={
                    "macro_f1": float(row_info["cv_macro_f1"]),
                    "balanced_accuracy": float(row_info["cv_balanced_accuracy"]),
                    "accuracy": float(row_info["cv_accuracy"]),
                },
            )
            parity = verify_svm_artifact_parity(X_va_raw, clf_obj, art)
            parity_checks[art_id] = parity

            pred_df = _prediction_rows_for_model(va_df, X_va_raw, art, model_variant=variant_name)
            prediction_frames.append(pred_df)

            cat_df = per_category_classification_table(y_va, pred_df["predicted_category"])
            cat_df.insert(0, "model_variant", variant_name)
            cat_df.insert(0, "feature_set", set_name)
            cat_df.insert(0, "method", primary_method)
            per_cat_frames.append(cat_df)

            if variant_name == "selected_best" and set_name in PRIMARY_FEATURE_SETS:
                canonical_id = f"svm_{primary_method}_{set_name}"
                art_canonical = dict(art)
                art_canonical["model_id"] = canonical_id
                model_artifacts[canonical_id] = art_canonical

    # Select overall primary model on k4_M0 across A, B, C
    primary_selected_rows = [
        r
        for r in summary_rows
        if r["track"] == "primary_k4_M0" and r["model_variant"] == "selected_best"
    ]
    overall_best_primary = max(primary_selected_rows, key=selection_key_for_svm)
    best_primary_set = str(overall_best_primary["feature_set"])
    selected_artifact = dict(model_artifacts[f"svm_{primary_method}_{best_primary_set}"])
    selected_artifact["model_id"] = f"svm_{primary_method}_selected"
    model_artifacts[f"svm_{primary_method}_selected"] = selected_artifact

    # Segmentation comparator sensitivity across otsu_M0, otsu_M2, k4_M0, k4_M1 for B and C
    comp_rows: list[dict[str, object]] = []
    primary_c_spec = next(
        r
        for r in primary_selected_rows
        if r["feature_set"] == "C"
    )
    for comp_method in comparators:
        tr_c, va_c, _ = load_classification_feature_frame(root, method=comp_method, segmented=False)
        for set_name in ("B", "C"):
            cols = feature_columns_for_set(set_name)
            X_tr_raw = tr_c[list(cols)].to_numpy(dtype=np.float64)
            y_tr = tr_c["category"].to_numpy(dtype=object)
            g_tr = tr_c["group_id"].to_numpy(dtype=object)
            X_va_raw = va_c[list(cols)].to_numpy(dtype=np.float64)
            y_va = va_c["category"].to_numpy(dtype=object)

            sc = fit_standard_scaler(X_tr_raw, cols)
            X_tr = apply_standard_scaler(X_tr_raw, sc)
            X_va = apply_standard_scaler(X_va_raw, sc)

            sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS)
            folds = []
            for f_tr_idx, f_va_idx in sgkf.split(X_tr_raw, y_tr, g_tr):
                sc_f = fit_standard_scaler(X_tr_raw[f_tr_idx], cols)
                folds.append(
                    (
                        f_va_idx,
                        apply_standard_scaler(X_tr_raw[f_tr_idx], sc_f),
                        y_tr[f_tr_idx],
                        apply_standard_scaler(X_tr_raw[f_va_idx], sc_f),
                    )
                )

            specs = [
                (
                    "default_rbf",
                    str(DEFAULT_BASELINE_SPEC["kernel"]),
                    float(DEFAULT_BASELINE_SPEC["C"]),
                    str(DEFAULT_BASELINE_SPEC["gamma"]),
                    str(DEFAULT_BASELINE_SPEC["class_weight"]),
                ),
                (
                    "primary_tuned_spec",
                    str(primary_c_spec["kernel"]),
                    float(primary_c_spec["C"]),
                    str(primary_c_spec["gamma"]),
                    str(primary_c_spec["class_weight"]),
                ),
            ]
            for spec_label, k_name, c_val, g_val, cw_val in specs:
                oof = np.empty_like(y_tr)
                for f_va_idx, Xf_tr, yf_tr, Xf_va in folds:
                    clf_f = fit_svm_model(
                        Xf_tr, yf_tr, kernel=k_name, C=c_val, gamma=g_val, class_weight=cw_val
                    )
                    oof[f_va_idx] = clf_f.predict(Xf_va)
                cv_m = classification_metrics(y_tr, oof)

                clf_c = fit_svm_model(
                    X_tr, y_tr, kernel=k_name, C=c_val, gamma=g_val, class_weight=cw_val
                )
                va_pred = clf_c.predict(X_va)
                va_m = classification_metrics(y_va, va_pred)
                comp_rows.append(
                    {
                        "method": comp_method,
                        "feature_set": set_name,
                        "n_features": len(cols),
                        "spec_variant": spec_label,
                        "kernel": k_name,
                        "class_weight": cw_val,
                        "C": c_val,
                        "gamma": g_val,
                        "cv_macro_f1": cv_m["macro_f1"],
                        "cv_balanced_accuracy": cv_m["balanced_accuracy"],
                        "cv_accuracy": cv_m["accuracy"],
                        "val_macro_f1": va_m["macro_f1"],
                        "val_balanced_accuracy": va_m["balanced_accuracy"],
                        "val_accuracy": va_m["accuracy"],
                        "val_weighted_f1": va_m["weighted_f1"],
                    }
                )

                if set_name == "C" and spec_label == "primary_tuned_spec" and comp_method != primary_method:
                    art_comp = export_svm_artifact(
                        model_id=f"svm_{comp_method}_C_primary_tuned_spec",
                        segmentation_method=comp_method,
                        feature_set="C",
                        scaler_payload=sc,
                        clf=clf_c,
                        split_sha256=split_sha,
                        validation_metrics=va_m,
                        cv_metrics=cv_m,
                    )
                    verify_svm_artifact_parity(X_va_raw, clf_c, art_comp)
                    prediction_frames.append(
                        _prediction_rows_for_model(
                            va_c, X_va_raw, art_comp, model_variant="comparator_tuned_spec"
                        )
                    )

    grid_results_df = pd.concat(grid_frames, ignore_index=True)
    selected_summary_df = pd.DataFrame(summary_rows)
    per_category_val_df = pd.concat(per_cat_frames, ignore_index=True)
    comparator_sensitivity_df = pd.DataFrame(comp_rows)
    val_predictions_df = pd.concat(prediction_frames, ignore_index=True)

    feature_dir = root / "results/features"
    input_artifact_hashes = {
        "split_manifest": sha256_bytes(root / "data/metadata/split_manifest.csv"),
        "feature_manifest": sha256_bytes(feature_dir / "feature_manifest.json"),
        "segmented_manifest": sha256_bytes(
            feature_dir / f"feature_manifest_segmented_{primary_method}.json"
        ),
        "appearance": sha256_bytes(feature_dir / "appearance.csv"),
        f"segmented_appearance_{primary_method}": sha256_bytes(
            feature_dir / f"appearance_segmented_{primary_method}.csv"
        ),
        f"features_{primary_method}_segmented": sha256_bytes(
            feature_dir / f"features_{primary_method}_segmented.csv"
        ),
        **{
            f"features_{m}": sha256_bytes(feature_dir / f"features_{m}.csv")
            for m in comparators
        },
    }

    manifest: dict[str, object] = {
        "schema_version": "svm-classification-v1",
        "primary_segmentation_method": primary_method,
        "comparators": list(comparators),
        "primary_feature_sets": list(PRIMARY_FEATURE_SETS),
        "sensitivity_feature_sets": list(SENSITIVITY_FEATURE_SETS),
        "split_sha256": split_sha,
        "cohort_counts": {
            "train_images": int(len(train_primary)),
            "train_groups": int(train_primary["group_id"].nunique()),
            "val_images": int(len(val_primary)),
            "val_groups": int(val_primary["group_id"].nunique()),
            "test_images_touched": 0,
        },
        "hyperparameter_grid": {
            "kernels": list(SVM_KERNELS),
            "class_weights": list(SVM_CLASS_WEIGHTS),
            "C_values": list(SVM_C_VALUES),
            "rbf_gammas": [_format_gamma(g) for g in SVM_RBF_GAMMAS],
            "cv_scheme": f"{CV_FOLDS}-fold StratifiedGroupKFold on train (group_id)",
        },
        "selection_rule": (
            "val_macro_f1 (4 dec) -> val_balanced_accuracy (4 dec) -> val_accuracy (4 dec) "
            "-> cv_macro_f1 (4 dec) -> simpler kernel (linear > rbf) -> smaller C"
        ),
        "selected_primary_model": {
            "model_id": f"svm_{primary_method}_selected",
            "source_model_id": f"svm_{primary_method}_{best_primary_set}",
            "segmentation_method": primary_method,
            "feature_set": best_primary_set,
            "n_features": int(overall_best_primary["n_features"]),
            "kernel": str(overall_best_primary["kernel"]),
            "class_weight": str(overall_best_primary["class_weight"]),
            "C": float(overall_best_primary["C"]),
            "gamma": str(overall_best_primary["gamma"]),
            "cv_macro_f1": float(overall_best_primary["cv_macro_f1"]),
            "cv_balanced_accuracy": float(overall_best_primary["cv_balanced_accuracy"]),
            "cv_accuracy": float(overall_best_primary["cv_accuracy"]),
            "val_macro_f1": float(overall_best_primary["val_macro_f1"]),
            "val_balanced_accuracy": float(overall_best_primary["val_balanced_accuracy"]),
            "val_accuracy": float(overall_best_primary["val_accuracy"]),
            "val_weighted_f1": float(overall_best_primary["val_weighted_f1"]),
        },
        "exported_model_files": [
            f"models/{model_id}.json" for model_id in sorted(model_artifacts)
        ],
        "artifact_parity_checks": parity_checks,
        "input_artifact_sha256": input_artifact_hashes,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    return SvmExperimentOutputs(
        grid_results=grid_results_df,
        selected_summary=selected_summary_df,
        per_category_val=per_category_val_df,
        comparator_sensitivity=comparator_sensitivity_df,
        val_predictions=val_predictions_df,
        model_artifacts=model_artifacts,
        manifest=manifest,
        split_sha256=split_sha,
    )


def load_svm_artifact(path: str | Path) -> dict[str, object]:
    """Load and schema-check an exported portable SVM JSON artifact."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing SVM artifact file: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "svm-classifier-v1":
        raise ValueError(
            f"Unsupported SVM artifact schema: {payload.get('schema_version')!r}."
        )
    required = {
        "model_id",
        "segmentation_method",
        "feature_set",
        "feature_names",
        "classes",
        "kernel",
        "C",
        "gamma_value",
        "scaler",
        "n_support",
        "support_vectors",
        "dual_coef",
        "intercept",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"SVM artifact missing required keys: {missing}.")
    return payload
