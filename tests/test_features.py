"""Tests for fixed-length appearance and mask-morphology feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ebhi_dip.dataset import load_split_manifest
from ebhi_dip.features import (
    APPEARANCE_FEATURE_NAMES,
    FEATURE_NAMES,
    MORPHOLOGY_FEATURE_NAMES,
    SEGMENTED_APPEARANCE_FEATURE_NAMES,
    SEGMENTED_FEATURE_NAMES,
    build_feature_tables,
    build_segmented_feature_tables,
    extract_appearance_features,
    extract_image_features,
    extract_masked_appearance_features,
    extract_morphology_features,
    feature_manifest,
    morphology_diagnostics,
    segmented_feature_manifest,
    validate_feature_table,
)
from ebhi_dip.paths import has_raw_dataset
from ebhi_dip.segmentation import predict_mask

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_appearance_feature_contract_and_flat_image_limits() -> None:
    assert len(APPEARANCE_FEATURE_NAMES) == 10
    assert len(FEATURE_NAMES) == 29
    assert len(MORPHOLOGY_FEATURE_NAMES) == 19

    flat = np.full((8, 8, 3), 128, dtype=np.uint8)
    features = extract_appearance_features(flat)
    assert tuple(features) == APPEARANCE_FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())
    assert features["lab_l_std"] == pytest.approx(0.0, abs=1e-12)
    assert features["glcm_contrast"] == pytest.approx(0.0, abs=1e-12)
    assert features["glcm_homogeneity"] == pytest.approx(1.0, abs=1e-12)
    assert features["glcm_energy"] == pytest.approx(1.0, abs=1e-12)
    assert features["glcm_correlation"] == pytest.approx(1.0, abs=1e-12)


def test_appearance_rejects_invalid_images() -> None:
    with pytest.raises(ValueError, match="uint8 RGB"):
        extract_appearance_features(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8 RGB"):
        extract_appearance_features(np.zeros((8, 8, 3), dtype=np.float32))


def test_masked_appearance_full_mask_matches_whole_image() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(24, 20, 3), dtype=np.uint8)
    whole = extract_appearance_features(image)
    masked = extract_masked_appearance_features(image, np.ones((24, 20), dtype=bool))
    assert tuple(masked) == SEGMENTED_APPEARANCE_FEATURE_NAMES
    for name, value in whole.items():
        assert masked[f"seg_{name}"] == pytest.approx(value, abs=1e-12)


def test_masked_appearance_excludes_background_and_handles_empty_mask() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 255
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:6] = True
    features = extract_masked_appearance_features(image, mask)
    assert tuple(features) == SEGMENTED_APPEARANCE_FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())
    assert features["seg_lab_l_mean"] == pytest.approx(53.2408, abs=1e-3)

    empty = extract_masked_appearance_features(image, np.zeros((8, 8), dtype=bool))
    assert all(value == 0.0 for value in empty.values())


def test_segmented_feature_names_are_distinct_and_complete() -> None:
    assert len(SEGMENTED_APPEARANCE_FEATURE_NAMES) == 10
    assert len(SEGMENTED_FEATURE_NAMES) == 29
    assert not set(SEGMENTED_APPEARANCE_FEATURE_NAMES) & set(APPEARANCE_FEATURE_NAMES)
    assert set(SEGMENTED_FEATURE_NAMES) == set(SEGMENTED_APPEARANCE_FEATURE_NAMES) | set(
        MORPHOLOGY_FEATURE_NAMES
    )


def test_morphology_square_metrics_and_edge_fraction() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    features = extract_morphology_features(mask)

    assert tuple(features) == MORPHOLOGY_FEATURE_NAMES
    assert features["mask_foreground_fraction"] == pytest.approx(16 / 100)
    assert features["mask_component_count"] == 1.0
    assert features["mask_edge_foreground_fraction"] == pytest.approx(0.0)
    assert features["mask_is_empty"] == 0.0
    assert features["region_area_rel_mean"] == pytest.approx(16 / 100)
    assert features["region_perimeter_rel_mean"] == pytest.approx(12 / 10)
    assert features["region_circularity_mean"] == pytest.approx(4 * np.pi * 16 / (12**2))
    assert features["region_eccentricity_mean"] == pytest.approx(0.0)
    assert features["region_solidity_mean"] == pytest.approx(1.0)

    border = np.zeros((10, 10), dtype=bool)
    border[0:2, 0:4] = True
    border_features = extract_morphology_features(border)
    assert border_features["mask_edge_foreground_fraction"] == pytest.approx(1.0)


def test_morphology_multiple_components_and_empty_policy() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 6:9] = True
    features = extract_morphology_features(mask)
    assert features["mask_component_count"] == 2.0
    areas = np.array([4 / 100, 9 / 100])
    assert features["region_area_rel_mean"] == pytest.approx(areas.mean())
    assert features["region_area_rel_std"] == pytest.approx(areas.std(ddof=0))
    assert features["region_area_rel_median"] == pytest.approx(np.median(areas))

    empty = extract_morphology_features(np.zeros((10, 10), dtype=bool))
    assert empty["mask_is_empty"] == 1.0
    assert all(
        value == 0.0 for name, value in empty.items() if name != "mask_is_empty"
    )
    assert morphology_diagnostics(np.zeros((10, 10), dtype=bool))["is_empty"] is True


def test_single_pixel_zero_perimeter_has_zero_circularity() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    features = extract_morphology_features(mask)
    assert features["region_circularity_mean"] == 0.0
    assert features["region_perimeter_rel_mean"] == 0.0
    diagnostics = morphology_diagnostics(mask)
    assert diagnostics["zero_perimeter_count"] == 1


def test_extract_image_features_concatenates_sets() -> None:
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    image[..., 1] = 90
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True
    combined = extract_image_features(image, mask)
    assert tuple(combined) == FEATURE_NAMES


def _valid_frame(count: int = 2) -> pd.DataFrame:
    rows = []
    for index in range(count):
        record = {
            "sample_id": f"s{index}",
            "category": "Normal",
            "group_id": "G1",
            "split": "val",
            "method": "k4_M0",
        }
        record.update({name: 0.5 for name in FEATURE_NAMES})
        record["mask_is_empty"] = 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def test_validate_feature_table_guards() -> None:
    frame = _valid_frame()
    assert validate_feature_table(frame) is frame

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_feature_table(duplicate)

    non_finite = frame.copy()
    non_finite.loc[0, "glcm_energy"] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_feature_table(non_finite)

    bad_category = frame.copy()
    bad_category.loc[0, "category"] = "Unknown"
    with pytest.raises(ValueError, match="unexpected categories"):
        validate_feature_table(bad_category)

    bad_fraction = frame.copy()
    bad_fraction.loc[0, "mask_foreground_fraction"] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_feature_table(bad_fraction)

    bad_flag = frame.copy()
    bad_flag.loc[0, "mask_is_empty"] = 0.5
    with pytest.raises(ValueError, match="binary"):
        validate_feature_table(bad_flag)

    missing = frame.drop(columns=["lab_l_mean"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_feature_table(missing)


@pytest.mark.skipif(not has_raw_dataset(REPO_ROOT), reason="EBHI-SEG raw dataset is unavailable.")
def test_build_feature_tables_parity_with_predict_mask() -> None:
    split_df = load_split_manifest(REPO_ROOT)
    sample = split_df.loc[split_df["split"].eq("val")].iloc[0]
    tables = build_feature_tables(REPO_ROOT, splits=("val",), sample_ids=[sample["sample_id"]])

    assert tables.appearance.shape == (1, 4 + 10)
    assert set(tables.combined) == {"otsu_M0", "otsu_M2", "k4_M0", "k4_M1"}
    assert tables.diagnostics["method"].nunique() == 4
    validate_feature_table(
        tables.appearance, feature_names=APPEARANCE_FEATURE_NAMES, require_method=False
    )
    for table in tables.combined.values():
        validate_feature_table(table)

    from ebhi_dip.dataset import load_rgb
    from ebhi_dip.paths import resolve_manifest_path

    image = load_rgb(resolve_manifest_path(REPO_ROOT, sample["image_path"]))
    for method in tables.methods:
        expected = extract_morphology_features(predict_mask(image, method, REPO_ROOT))
        row = tables.combined[method].iloc[0]
        for name, value in expected.items():
            assert row[name] == pytest.approx(value, abs=1e-12)

    manifest = feature_manifest(tables, REPO_ROOT)
    assert manifest["schema_version"] == "feature-extraction-v1"
    assert manifest["split_sha256"] == tables.split_sha256
    assert manifest["sample_counts"] == {"val": 1}


@pytest.mark.skipif(not has_raw_dataset(REPO_ROOT), reason="EBHI-SEG raw dataset is unavailable.")
def test_build_segmented_feature_tables_parity_with_predict_mask() -> None:
    split_df = load_split_manifest(REPO_ROOT)
    sample = split_df.loc[split_df["split"].eq("val")].iloc[0]
    tables = build_segmented_feature_tables(
        REPO_ROOT,
        method="k4_M0",
        splits=("val",),
        sample_ids=[sample["sample_id"]],
    )

    assert tables.appearance.shape == (1, 4 + 10)
    assert tables.combined.shape == (1, 5 + 10 + 19)
    assert tables.diagnostics.shape[0] == 1
    validate_feature_table(
        tables.appearance,
        feature_names=SEGMENTED_APPEARANCE_FEATURE_NAMES,
        require_method=False,
    )
    validate_feature_table(tables.combined, feature_names=SEGMENTED_FEATURE_NAMES)

    from ebhi_dip.dataset import load_rgb
    from ebhi_dip.paths import resolve_manifest_path

    image = load_rgb(resolve_manifest_path(REPO_ROOT, sample["image_path"]))
    mask = predict_mask(image, "k4_M0", REPO_ROOT)
    expected = extract_masked_appearance_features(image, mask)
    row = tables.appearance.iloc[0]
    for name, value in expected.items():
        assert row[name] == pytest.approx(value, abs=1e-12)

    manifest = segmented_feature_manifest(tables, REPO_ROOT)
    assert manifest["schema_version"] == "segmented-feature-extraction-v1"
    assert manifest["source_method"] == "k4_M0"
    assert manifest["split_sha256"] == tables.split_sha256
    assert manifest["sample_counts"] == {"val": 1}
