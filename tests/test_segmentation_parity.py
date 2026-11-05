"""Tests for segmentation numerical contracts, morphology edge cases, and image inference."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ebhi_dip.dataset import load_reference_mask, load_split_manifest, load_rgb
from ebhi_dip.paths import has_raw_dataset
from ebhi_dip.segmentation import (
    SUPPORTED_METHODS,
    morphology_variants,
    nearest_clusters,
    predict_mask,
    rgb_to_lab,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_nearest_clusters_float64_and_lowest_id_tie_break() -> None:
    centers = np.array(
        [
            [50.0, 0.0, 0.0],
            [50.0, 10.0, 0.0],
            [50.0, 0.0, 10.0],
        ],
        dtype=np.float32,
    )
    pixels = np.array(
        [
            [50.0, 5.0, 0.0],   # Exact tie between cluster 0 (dist^2=25) and cluster 1 (dist^2=25) -> 0
            [50.0, 6.0, 6.0],   # dist^2: c0=72, c1=52, c2=52 -> tie between 1 and 2 -> 1
            [50.0, 0.0, 9.0],   # Closest to cluster 2 -> 2
        ],
        dtype=np.float32,
    )
    labels = nearest_clusters(pixels, centers, chunk_size=2)
    assert labels.dtype == np.int32
    assert labels.tolist() == [0, 1, 2]


def test_reference_mask_threshold_127_vs_128(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    arr = np.array([[0, 127], [128, 255]], dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(mask_path)

    binary = load_reference_mask(mask_path, threshold=128)
    assert binary.dtype == np.bool_
    assert np.array_equal(binary, np.array([[False, False], [True, True]]))


def test_morphology_variants_area_connectivity_and_border_replicate() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    # Component A: 15 pixels (3x5) -> must be removed in M1
    mask[2:5, 2:7] = True
    # Component B: 16 pixels connected only via an 8-connected diagonal bridge!
    mask[10:12, 2:6] = True   # 8 pixels
    mask[12:14, 6:10] = True  # 8 pixels touching (11,5)-(12,6) diagonally -> total 16 under 8-conn!

    variants = morphology_variants(mask)
    assert np.count_nonzero(variants["M0"]) == 31
    assert np.count_nonzero(variants["M1"][2:5, 2:7]) == 0
    assert np.count_nonzero(variants["M1"][10:14, 2:10]) == 16

    # Border behavior: 4x4 block flush against top-left border (0:4, 0:4), area=16.
    # Under BORDER_REPLICATE with 3x3 cross opening, border corners (0,0), (0,3), (3,0)
    # survive opening while interior convex corner (3,3) is rounded off (15 pixels kept).
    border_mask = np.zeros((12, 12), dtype=bool)
    border_mask[0:4, 0:4] = True
    b_vars = morphology_variants(border_mask)
    assert bool(b_vars["M2"][0, 0]) is True
    assert bool(b_vars["M2"][0, 3]) is True
    assert bool(b_vars["M2"][3, 0]) is True
    assert bool(b_vars["M2"][3, 3]) is False
    assert np.count_nonzero(b_vars["M2"][0:4, 0:4]) == 15


def test_predict_mask_contract_and_raw_image_smoke() -> None:
    sig = inspect.signature(predict_mask)
    assert list(sig.parameters) == ["image_rgb", "method_id", "project_root"]

    synthetic = np.zeros((16, 16, 3), dtype=np.uint8)
    synthetic[:8, :, :] = 40
    synthetic[8:, :, :] = 220

    for method_id in SUPPORTED_METHODS:
        pred = predict_mask(synthetic, method_id, REPO_ROOT)
        assert pred.shape == (16, 16)
        assert pred.dtype == np.bool_

    lab = rgb_to_lab(synthetic)
    assert lab.dtype == np.float32
    assert lab.shape == (16, 16, 3)

    if has_raw_dataset(REPO_ROOT):
        split_df = load_split_manifest(REPO_ROOT)
        train_row = split_df.loc[split_df["split"].eq("train")].iloc[0]
        img = load_rgb(REPO_ROOT / train_row["image_path"])
        mask_k4 = predict_mask(img, "k4_M0", REPO_ROOT)
        assert mask_k4.shape == (224, 224)
        assert 0 < int(np.count_nonzero(mask_k4)) < 224 * 224
