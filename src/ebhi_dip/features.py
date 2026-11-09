"""Fixed-length appearance (color/GLCM) and mask-morphology feature extraction.

Feature set A (10): Lab channel mean/population-std and four averaged GLCM properties.
Feature set B (19): four global mask descriptors plus mean/std/median of five region measures.
Feature set C (29): concatenation A + B.
Segmented sets A_seg (10) and C_seg (29): appearance computed inside one predicted mask.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy
import skimage
from skimage.feature import graycoprops, graycomatrix
from skimage.measure import label, regionprops

from ebhi_dip.dataset import EXPECTED_CATEGORIES, load_rgb, load_split_manifest
from ebhi_dip.paths import resolve_manifest_path, verify_frozen_split
from ebhi_dip.segmentation import (
    SUPPORTED_METHODS,
    _verify_postprocessing_provenance,
    load_kmeans_model,
    load_otsu_config,
    morphology_variants,
    nearest_clusters,
    otsu_threshold_mask,
    rgb_to_gray,
    rgb_to_lab,
)

APPEARANCE_FEATURE_NAMES: tuple[str, ...] = (
    "lab_l_mean",
    "lab_l_std",
    "lab_a_mean",
    "lab_a_std",
    "lab_b_mean",
    "lab_b_std",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
)

MORPHOLOGY_FEATURE_NAMES: tuple[str, ...] = (
    "mask_foreground_fraction",
    "mask_component_count",
    "mask_edge_foreground_fraction",
    "mask_is_empty",
    "region_area_rel_mean",
    "region_area_rel_std",
    "region_area_rel_median",
    "region_perimeter_rel_mean",
    "region_perimeter_rel_std",
    "region_perimeter_rel_median",
    "region_circularity_mean",
    "region_circularity_std",
    "region_circularity_median",
    "region_eccentricity_mean",
    "region_eccentricity_std",
    "region_eccentricity_median",
    "region_solidity_mean",
    "region_solidity_std",
    "region_solidity_median",
)

FEATURE_NAMES: tuple[str, ...] = APPEARANCE_FEATURE_NAMES + MORPHOLOGY_FEATURE_NAMES
SEGMENTED_APPEARANCE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"seg_{name}" for name in APPEARANCE_FEATURE_NAMES
)
SEGMENTED_FEATURE_NAMES: tuple[str, ...] = (
    SEGMENTED_APPEARANCE_FEATURE_NAMES + MORPHOLOGY_FEATURE_NAMES
)

GLCM_LEVELS: int = 16
GLCM_DISTANCES: tuple[int, ...] = (1, 2)
GLCM_ANGLES: tuple[float, ...] = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
GLCM_PROPERTIES: tuple[str, ...] = ("contrast", "homogeneity", "energy", "correlation")

DEFAULT_FEATURE_METHODS: tuple[str, ...] = ("otsu_M0", "otsu_M2", "k4_M0", "k4_M1")
PRIMARY_FEATURE_METHOD: str = "k4_M0"
DEFAULT_FEATURE_SPLITS: tuple[str, ...] = ("train", "val", "test")

METADATA_COLUMNS: tuple[str, ...] = ("sample_id", "category", "group_id", "split")

_REGION_MEASURES: tuple[str, ...] = (
    "area_rel",
    "perimeter_rel",
    "circularity",
    "eccentricity",
    "solidity",
)


@dataclass(frozen=True)
class FeatureTables:
    """Container for extraction outputs across the frozen cohort."""

    appearance: pd.DataFrame
    combined: dict[str, pd.DataFrame]
    diagnostics: pd.DataFrame
    methods: tuple[str, ...]
    splits: tuple[str, ...]
    split_sha256: str


@dataclass(frozen=True)
class SegmentedFeatureTables:
    """Container for appearance features restricted to a frozen predicted mask."""

    appearance: pd.DataFrame
    combined: pd.DataFrame
    diagnostics: pd.DataFrame
    method: str
    splits: tuple[str, ...]
    split_sha256: str


def _validate_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image_rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"Expected uint8 RGB image of shape (H, W, 3), got dtype={rgb.dtype}, shape={rgb.shape}."
        )
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("RGB image dimensions must be non-zero.")
    return rgb


def _validate_binary_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D binary mask, got shape {arr.shape}.")
    if arr.dtype != np.bool_:
        if not np.issubdtype(arr.dtype, np.number) or not np.isin(arr, (0, 1)).all():
            raise ValueError(f"Mask must be boolean or binary {{0, 1}}, got dtype {arr.dtype}.")
        arr = arr.astype(bool)
    return arr


def _quantize_gray(gray: np.ndarray) -> np.ndarray:
    """Quantize uint8 grayscale to the fixed GLCM level range."""
    return np.clip(
        gray.astype(np.int32) // (256 // GLCM_LEVELS),
        0,
        GLCM_LEVELS - 1,
    ).astype(np.uint8)


def _glcm_properties(matrix: np.ndarray) -> dict[str, float]:
    """Compute the four averaged GLCM properties with finite fallbacks."""
    features: dict[str, float] = {}
    for prop in GLCM_PROPERTIES:
        value = float(np.mean(graycoprops(matrix, prop)))
        features[f"glcm_{prop}"] = value if np.isfinite(value) else 0.0
    return features


def _masked_glcm(quantized: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build a normalized GLCM from pairs whose two endpoints are inside the mask.

    A sentinel level is assigned to background pixels. The sentinel row and column are
    then discarded, leaving only foreground-foreground pairs. This preserves the exact
    offset conventions and symmetric/normalized behavior of ``skimage.graycomatrix``.
    """
    sentinel = GLCM_LEVELS
    background_level = np.uint8(sentinel)
    masked = np.where(mask, quantized, background_level).astype(np.uint8)
    matrix = graycomatrix(
        masked,
        distances=list(GLCM_DISTANCES),
        angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS + 1,
        symmetric=True,
        normed=False,
    )
    foreground = matrix[:GLCM_LEVELS, :GLCM_LEVELS, :, :].astype(np.float64)
    sums = foreground.sum(axis=(0, 1), keepdims=True)
    sums[sums == 0.0] = 1.0
    return foreground / sums


def _appearance_from_lab_and_glcm(
    lab: np.ndarray,
    glcm_features: dict[str, float],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for channel_index, channel_name in enumerate(("l", "a", "b")):
        values = lab[..., channel_index].astype(np.float64)
        features[f"lab_{channel_name}_mean"] = float(values.mean())
        features[f"lab_{channel_name}_std"] = float(values.std(ddof=0))
    features.update(glcm_features)
    return {name: float(features[name]) for name in APPEARANCE_FEATURE_NAMES}


def extract_appearance_features(image_rgb: np.ndarray) -> dict[str, float]:
    """Compute the 10 appearance features from the original RGB image."""
    rgb = _validate_rgb_image(image_rgb)
    lab = rgb_to_lab(rgb)
    gray = rgb_to_gray(rgb)
    quantized = _quantize_gray(gray)
    matrix = graycomatrix(
        quantized,
        distances=list(GLCM_DISTANCES),
        angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    return _appearance_from_lab_and_glcm(lab, _glcm_properties(matrix))


def extract_masked_appearance_features(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute appearance features using only pixels inside a predicted binary mask."""
    rgb = _validate_rgb_image(image_rgb)
    mask_arr = _validate_binary_mask(mask)
    if mask_arr.shape != rgb.shape[:2]:
        raise ValueError(
            f"Mask shape {mask_arr.shape} does not match image shape {rgb.shape[:2]}."
        )

    features = {name: 0.0 for name in SEGMENTED_APPEARANCE_FEATURE_NAMES}
    if not mask_arr.any():
        return features

    lab = rgb_to_lab(rgb)
    foreground_lab = lab[mask_arr].astype(np.float64)

    gray = rgb_to_gray(rgb)
    quantized = _quantize_gray(gray)
    matrix = _masked_glcm(quantized, mask_arr)
    glcm_features = _glcm_properties(matrix)

    base = _appearance_from_lab_and_glcm(foreground_lab, glcm_features)
    for name, value in base.items():
        features[f"seg_{name}"] = value
    return {name: float(features[name]) for name in SEGMENTED_APPEARANCE_FEATURE_NAMES}


def _region_measurements(mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Return per-region measurements and mask-level diagnostics for a non-empty mask."""
    height, width = mask.shape
    image_area = float(height * width)
    image_diagonal_scale = float(np.sqrt(image_area))

    labels = label(mask, connectivity=2)
    properties = regionprops(labels)

    areas = np.asarray([prop.area for prop in properties], dtype=np.float64)
    perimeters = np.asarray([prop.perimeter for prop in properties], dtype=np.float64)
    eccentricities = np.asarray([prop.eccentricity for prop in properties], dtype=np.float64)
    solidities = np.asarray([prop.solidity for prop in properties], dtype=np.float64)

    circularities = np.zeros_like(areas)
    positive_perimeter = perimeters > 0.0
    circularities[positive_perimeter] = (
        4.0 * np.pi * areas[positive_perimeter] / np.square(perimeters[positive_perimeter])
    )

    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    border_labels = border_labels[border_labels != 0]
    border_set = set(int(value) for value in border_labels)
    edge_foreground = int(
        sum(prop.area for prop in properties if int(prop.label) in border_set)
    )

    measurements = {
        "area_rel": areas / image_area,
        "perimeter_rel": perimeters / image_diagonal_scale,
        "circularity": circularities,
        "eccentricity": eccentricities,
        "solidity": solidities,
    }
    diagnostics = {
        "is_empty": False,
        "component_count": int(len(properties)),
        "zero_perimeter_count": int(np.count_nonzero(~positive_perimeter)),
        "border_component_count": int(len(border_set)),
        "foreground_pixels": int(mask.sum()),
        "edge_foreground_pixels": edge_foreground,
        "image_pixels": int(image_area),
    }
    return measurements, diagnostics


def _morphology_payload(mask: np.ndarray) -> tuple[dict[str, float], dict[str, object]]:
    """Compute morphology features and mask-level diagnostics from a single region pass."""
    arr = _validate_binary_mask(mask)
    height, width = arr.shape
    image_area = float(height * width)
    foreground_pixels = int(arr.sum())

    if foreground_pixels == 0:
        features = {
            "mask_foreground_fraction": 0.0,
            "mask_component_count": 0.0,
            "mask_edge_foreground_fraction": 0.0,
            "mask_is_empty": 1.0,
        }
        for measure in _REGION_MEASURES:
            features[f"region_{measure}_mean"] = 0.0
            features[f"region_{measure}_std"] = 0.0
            features[f"region_{measure}_median"] = 0.0
        diagnostics = {
            "is_empty": True,
            "component_count": 0,
            "zero_perimeter_count": 0,
            "border_component_count": 0,
            "foreground_pixels": 0,
            "edge_foreground_pixels": 0,
            "image_pixels": int(height * width),
        }
        return {name: float(features[name]) for name in MORPHOLOGY_FEATURE_NAMES}, diagnostics

    measurements, diagnostics = _region_measurements(arr)
    edge_fraction = float(diagnostics["edge_foreground_pixels"]) / float(foreground_pixels)
    features = {
        "mask_foreground_fraction": float(foreground_pixels) / image_area,
        "mask_component_count": float(diagnostics["component_count"]),
        "mask_edge_foreground_fraction": edge_fraction,
        "mask_is_empty": 0.0,
    }
    for measure in _REGION_MEASURES:
        values = measurements[measure]
        features[f"region_{measure}_mean"] = float(np.mean(values))
        features[f"region_{measure}_std"] = float(np.std(values, ddof=0))
        features[f"region_{measure}_median"] = float(np.median(values))

    ordered = {name: float(features[name]) for name in MORPHOLOGY_FEATURE_NAMES}
    return ordered, diagnostics


def morphology_diagnostics(mask: np.ndarray) -> dict[str, object]:
    """Return mask-level diagnostics used to audit empty and degenerate regions."""
    _, diagnostics = _morphology_payload(mask)
    return diagnostics


def extract_morphology_features(mask: np.ndarray) -> dict[str, float]:
    """Compute the 19 mask morphology features from a predicted binary mask."""
    features, _ = _morphology_payload(mask)
    return features


def extract_image_features(image_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute the combined 29-feature vector for one image and one predicted mask."""
    appearance = extract_appearance_features(image_rgb)
    morphology = extract_morphology_features(mask)
    return {**appearance, **morphology}


def _family_and_stage(method: str) -> tuple[str, str]:
    family, _, stage = method.partition("_")
    return family, stage


def _load_method_artifacts(project_root: Path, methods: Sequence[str]) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    groups = {_family_and_stage(method)[0] for method in methods}
    for family in sorted(groups):
        if family == "otsu":
            artifacts["otsu_config"] = load_otsu_config(project_root)
        elif family == "all":
            artifacts["all_foreground"] = None
        elif family.startswith("k") and family[1:].isdigit():
            artifacts[f"k{int(family[1:])}"] = load_kmeans_model(
                project_root, k=int(family[1:]), verify_provenance=True
            )
        else:
            raise ValueError(f"Unsupported segmentation family '{family}'.")
    return artifacts


def _predict_variants(
    image_rgb: np.ndarray,
    artifacts: dict[str, object],
) -> dict[str, np.ndarray]:
    """Compute all morphology stages per requested family with preloaded artifacts."""
    variants: dict[str, np.ndarray] = {}
    if "otsu_config" in artifacts:
        config = artifacts["otsu_config"]
        _, raw = otsu_threshold_mask(image_rgb, str(config["foreground_polarity"]))
        variants.update({f"otsu_{stage}": mask for stage, mask in morphology_variants(raw).items()})

    for key, model in artifacts.items():
        if not key.startswith("k"):
            continue
        if not isinstance(model, dict):
            raise TypeError(f"Expected a K-means model mapping for '{key}'.")
        k = int(key[1:])
        lab = rgb_to_lab(image_rgb)
        labels = nearest_clusters(lab.reshape(-1, 3), model["cluster_centers"]).reshape(
            image_rgb.shape[:2]
        )
        raw = np.isin(labels, model["foreground_clusters"])
        variants.update({f"k{k}_{stage}": mask for stage, mask in morphology_variants(raw).items()})

    if "all_foreground" in artifacts:
        variants["all_foreground"] = np.ones(image_rgb.shape[:2], dtype=bool)

    return variants


_WORKER_STATE: dict[str, object] = {}


def _init_worker(
    project_root: str,
    data_root_override: str | None,
    methods: tuple[str, ...],
) -> None:
    root = Path(project_root)
    artifacts = _load_method_artifacts(root, methods)
    if any(_family_and_stage(method)[1] != "M0" for method in methods):
        _verify_postprocessing_provenance(root)
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        project_root=root,
        data_root_override=data_root_override,
        methods=methods,
        artifacts=artifacts,
    )


def _process_record(
    record: dict[str, object],
) -> tuple[
    dict[str, object],
    dict[str, float],
    dict[str, dict[str, float]],
    dict[str, dict[str, object]],
]:
    root = _WORKER_STATE["project_root"]
    methods = _WORKER_STATE["methods"]
    artifacts = _WORKER_STATE["artifacts"]
    if not isinstance(root, Path) or not isinstance(methods, tuple) or not isinstance(artifacts, dict):
        raise RuntimeError("Feature extraction worker was not initialized.")

    image_path = resolve_manifest_path(
        root,
        str(record["image_path"]),
        data_root_override=_WORKER_STATE["data_root_override"],
    )
    image_rgb = load_rgb(image_path)
    appearance = extract_appearance_features(image_rgb)
    metadata = {
        "sample_id": str(record["sample_id"]),
        "category": str(record["category"]),
        "group_id": str(record["group_id"]),
        "split": str(record["split"]),
    }

    variants = _predict_variants(image_rgb, artifacts)
    morphologies: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    for method in methods:
        morphologies[method], diagnostics[method] = _morphology_payload(variants[method])

    return metadata, appearance, morphologies, diagnostics


def _process_segmented_record(
    record: dict[str, object],
) -> tuple[
    dict[str, object],
    dict[str, float],
    dict[str, float],
    dict[str, object],
]:
    root = _WORKER_STATE["project_root"]
    methods = _WORKER_STATE["methods"]
    artifacts = _WORKER_STATE["artifacts"]
    if not isinstance(root, Path) or not isinstance(methods, tuple) or not isinstance(artifacts, dict):
        raise RuntimeError("Segmented feature extraction worker was not initialized.")
    if len(methods) != 1:
        raise RuntimeError("Segmented feature extraction requires exactly one method.")
    method = methods[0]

    image_path = resolve_manifest_path(
        root,
        str(record["image_path"]),
        data_root_override=_WORKER_STATE["data_root_override"],
    )
    image_rgb = load_rgb(image_path)
    metadata = {
        "sample_id": str(record["sample_id"]),
        "category": str(record["category"]),
        "group_id": str(record["group_id"]),
        "split": str(record["split"]),
    }
    mask = _predict_variants(image_rgb, artifacts)[method]
    appearance = extract_masked_appearance_features(image_rgb, mask)
    morphology, diagnostics = _morphology_payload(mask)
    return metadata, appearance, morphology, diagnostics


def _select_feature_subset(
    root: Path,
    splits: Sequence[str],
    sample_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    split_tuple = tuple(splits)
    if not split_tuple:
        raise ValueError("At least one split is required.")
    split_df = load_split_manifest(root, verify=False)
    unknown_splits = sorted(set(split_tuple) - set(split_df["split"].unique()))
    if unknown_splits:
        raise ValueError(f"Unknown split labels: {unknown_splits}.")

    subset = split_df.loc[split_df["split"].isin(split_tuple)].copy()
    if sample_ids is not None:
        requested = list(dict.fromkeys(str(value) for value in sample_ids))
        subset = subset.loc[subset["sample_id"].isin(requested)]
        missing = sorted(set(requested) - set(subset["sample_id"]))
        if missing:
            raise KeyError(f"Requested sample_ids not found in the frozen split: {missing[:5]}.")
    subset = subset.sort_values(
        ["split", "sample_id"], ascending=True, kind="mergesort"
    ).reset_index(drop=True)
    if not subset["pair_status"].eq("paired").all():
        raise ValueError("Feature extraction requires paired image/reference rows.")
    return split_tuple, subset


def build_feature_tables(
    project_root: str | Path,
    methods: Sequence[str] = DEFAULT_FEATURE_METHODS,
    splits: Sequence[str] = DEFAULT_FEATURE_SPLITS,
    data_root_override: str | Path | None = None,
    sample_ids: Sequence[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
) -> FeatureTables:
    """Extract A and A+B feature tables for every paired image in the selected splits."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)

    method_tuple = tuple(methods)
    if not method_tuple:
        raise ValueError("At least one segmentation method is required.")
    unknown_methods = sorted(set(method_tuple) - set(SUPPORTED_METHODS))
    if unknown_methods:
        raise ValueError(f"Unsupported segmentation methods: {unknown_methods}.")

    split_tuple, subset = _select_feature_subset(root, splits, sample_ids)

    if int(workers) < 1:
        raise ValueError("workers must be at least 1.")
    total = len(subset)
    records = subset.to_dict("records")
    data_root_arg = None if data_root_override is None else str(data_root_override)

    appearance_rows: list[dict[str, object]] = []
    morphology_rows: dict[str, list[dict[str, object]]] = {method: [] for method in method_tuple}
    diagnostics_rows: list[dict[str, object]] = []

    if int(workers) == 1:
        _init_worker(str(root), data_root_arg, method_tuple)
        results = map(_process_record, records)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_init_worker,
            initargs=(str(root), data_root_arg, method_tuple),
        )
        results = executor.map(
            _process_record,
            records,
            chunksize=max(1, total // (int(workers) * 8)),
        )

    try:
        for position, (metadata, appearance, morphologies, diagnostics) in enumerate(results, start=1):
            appearance_rows.append({**metadata, **appearance})
            for method in method_tuple:
                morphology_rows[method].append({**metadata, "method": method, **morphologies[method]})
                diagnostics_rows.append({**metadata, "method": method, **diagnostics[method]})
            if progress is not None:
                progress(position, total)
    finally:
        if executor is not None:
            executor.shutdown()
        _WORKER_STATE.clear()

    appearance_df = pd.DataFrame(appearance_rows, columns=[*METADATA_COLUMNS, *APPEARANCE_FEATURE_NAMES])

    combined: dict[str, pd.DataFrame] = {}
    for method in method_tuple:
        method_df = pd.DataFrame(
            morphology_rows[method],
            columns=[*METADATA_COLUMNS, "method", *MORPHOLOGY_FEATURE_NAMES],
        )
        merged = appearance_df.merge(
            method_df,
            on=list(METADATA_COLUMNS),
            how="inner",
            validate="one_to_one",
        )
        combined[method] = merged[
            [*METADATA_COLUMNS, "method", *APPEARANCE_FEATURE_NAMES, *MORPHOLOGY_FEATURE_NAMES]
        ]

    diagnostics_columns = [
        *METADATA_COLUMNS,
        "method",
        "is_empty",
        "component_count",
        "zero_perimeter_count",
        "border_component_count",
        "foreground_pixels",
        "edge_foreground_pixels",
        "image_pixels",
    ]
    diagnostics_df = pd.DataFrame(diagnostics_rows, columns=diagnostics_columns)

    return FeatureTables(
        appearance=appearance_df,
        combined=combined,
        diagnostics=diagnostics_df,
        methods=method_tuple,
        splits=split_tuple,
        split_sha256=split_sha,
    )


def build_segmented_feature_tables(
    project_root: str | Path,
    method: str = PRIMARY_FEATURE_METHOD,
    splits: Sequence[str] = DEFAULT_FEATURE_SPLITS,
    data_root_override: str | Path | None = None,
    sample_ids: Sequence[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
) -> SegmentedFeatureTables:
    """Extract A_seg and A_seg+B features using one frozen predicted mask method."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported segmentation method: {method!r}.")
    split_tuple, subset = _select_feature_subset(root, splits, sample_ids)

    if int(workers) < 1:
        raise ValueError("workers must be at least 1.")
    total = len(subset)
    records = subset.to_dict("records")
    method_tuple = (method,)
    data_root_arg = None if data_root_override is None else str(data_root_override)

    appearance_rows: list[dict[str, object]] = []
    morphology_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []

    if int(workers) == 1:
        _init_worker(str(root), data_root_arg, method_tuple)
        results = map(_process_segmented_record, records)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_init_worker,
            initargs=(str(root), data_root_arg, method_tuple),
        )
        results = executor.map(
            _process_segmented_record,
            records,
            chunksize=max(1, total // (int(workers) * 8)),
        )

    try:
        for position, (metadata, appearance, morphology, diagnostics) in enumerate(
            results, start=1
        ):
            appearance_rows.append({**metadata, **appearance})
            morphology_rows.append({**metadata, "method": method, **morphology})
            diagnostics_rows.append({**metadata, "method": method, **diagnostics})
            if progress is not None:
                progress(position, total)
    finally:
        if executor is not None:
            executor.shutdown()
        _WORKER_STATE.clear()

    appearance_df = pd.DataFrame(
        appearance_rows,
        columns=[*METADATA_COLUMNS, *SEGMENTED_APPEARANCE_FEATURE_NAMES],
    )
    morphology_df = pd.DataFrame(
        morphology_rows,
        columns=[*METADATA_COLUMNS, "method", *MORPHOLOGY_FEATURE_NAMES],
    )
    combined = appearance_df.merge(
        morphology_df,
        on=list(METADATA_COLUMNS),
        how="inner",
        validate="one_to_one",
    )[
        [
            *METADATA_COLUMNS,
            "method",
            *SEGMENTED_APPEARANCE_FEATURE_NAMES,
            *MORPHOLOGY_FEATURE_NAMES,
        ]
    ]

    diagnostics_columns = [
        *METADATA_COLUMNS,
        "method",
        "is_empty",
        "component_count",
        "zero_perimeter_count",
        "border_component_count",
        "foreground_pixels",
        "edge_foreground_pixels",
        "image_pixels",
    ]
    diagnostics_df = pd.DataFrame(diagnostics_rows, columns=diagnostics_columns)

    return SegmentedFeatureTables(
        appearance=appearance_df,
        combined=combined,
        diagnostics=diagnostics_df,
        method=method,
        splits=split_tuple,
        split_sha256=split_sha,
    )


def validate_feature_table(
    table: pd.DataFrame,
    feature_names: Sequence[str] = FEATURE_NAMES,
    require_method: bool = True,
    expected_categories: Sequence[str] = EXPECTED_CATEGORIES,
) -> pd.DataFrame:
    """Validate metadata, dimensions, uniqueness, finiteness, and bounded feature ranges."""
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Feature table must be a non-empty DataFrame.")

    names = tuple(feature_names)
    required = set(METADATA_COLUMNS) | set(names)
    if require_method:
        required.add("method")
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}.")

    if table["sample_id"].isna().any() or table["category"].isna().any():
        raise ValueError("Feature table contains missing sample_id or category values.")
    keys = ["sample_id", "method"] if require_method else ["sample_id"]
    if table.duplicated(keys).any():
        raise ValueError(f"Feature table contains duplicate rows for key {keys}.")

    values = table[list(names)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Feature table contains NaN or infinite feature values.")

    unexpected_categories = sorted(set(table["category"]) - set(expected_categories))
    if unexpected_categories:
        raise ValueError(f"Feature table contains unexpected categories: {unexpected_categories}.")

    for column in ("mask_foreground_fraction", "mask_edge_foreground_fraction"):
        if column in table.columns:
            series = table[column].to_numpy(dtype=np.float64)
            if np.any((series < 0.0) | (series > 1.0)):
                raise ValueError(f"Feature '{column}' must lie in [0, 1].")
    if "mask_is_empty" in table.columns:
        series = table["mask_is_empty"].to_numpy(dtype=np.float64)
        if not np.isin(series, (0.0, 1.0)).all():
            raise ValueError("Feature 'mask_is_empty' must be binary {0, 1}.")

    return table


def feature_manifest(
    tables: FeatureTables,
    project_root: str | Path,
    primary_method: str = PRIMARY_FEATURE_METHOD,
) -> dict[str, object]:
    """Build a JSON-serializable manifest documenting feature definitions and diagnostics."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    if split_sha != tables.split_sha256:
        raise ValueError("Frozen split checksum changed after feature extraction.")

    diagnostics = tables.diagnostics
    empty_by_method = (
        diagnostics.groupby("method", observed=True)["is_empty"].sum().astype(int).to_dict()
        if not diagnostics.empty
        else {}
    )
    zero_perimeter_by_method = (
        diagnostics.groupby("method", observed=True)["zero_perimeter_count"]
        .sum()
        .astype(int)
        .to_dict()
        if not diagnostics.empty
        else {}
    )
    sample_counts = tables.appearance.groupby("split", observed=True).size().astype(int).to_dict()

    return {
        "schema_version": "feature-extraction-v1",
        "primary_method": primary_method,
        "methods": list(tables.methods),
        "splits": list(tables.splits),
        "split_sha256": split_sha,
        "feature_sets": {
            "A": list(APPEARANCE_FEATURE_NAMES),
            "B": list(MORPHOLOGY_FEATURE_NAMES),
            "C": list(FEATURE_NAMES),
        },
        "appearance_definition": {
            "color_space": "CIE Lab via OpenCV RGB float32/255 -> COLOR_RGB2LAB",
            "lab_summary": "mean and population standard deviation over all pixels",
            "glcm_levels": GLCM_LEVELS,
            "glcm_distances": list(GLCM_DISTANCES),
            "glcm_angles_radians": list(GLCM_ANGLES),
            "glcm_quantization": "fixed interval gray // 16 over full 0-255 range",
            "glcm_symmetric": True,
            "glcm_normalized": True,
            "glcm_aggregation": "mean over the eight distance/angle pairs",
        },
        "morphology_definition": {
            "connectivity": 8,
            "edge_fraction": "foreground pixels in border-touching components / total foreground",
            "area_normalization": "component area / image area",
            "perimeter_normalization": "component perimeter / sqrt(image area)",
            "circularity": "4*pi*area / perimeter^2; zero when perimeter is zero",
            "eccentricity": "skimage.measure.regionprops eccentricity",
            "solidity": "skimage.measure.regionprops solidity",
            "summaries": ["mean", "population_std", "median"],
            "empty_mask_policy": "all numeric features zero, mask_is_empty=1",
        },
        "sample_counts": sample_counts,
        "empty_masks_by_method": empty_by_method,
        "zero_perimeter_components_by_method": zero_perimeter_by_method,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
        },
    }


def segmented_feature_manifest(
    tables: SegmentedFeatureTables,
    project_root: str | Path,
) -> dict[str, object]:
    """Build a JSON-serializable manifest for mask-restricted appearance features."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    if split_sha != tables.split_sha256:
        raise ValueError("Frozen split checksum changed after segmented feature extraction.")

    diagnostics = tables.diagnostics
    empty_masks = int(diagnostics["is_empty"].sum())
    zero_perimeter = int(diagnostics["zero_perimeter_count"].sum())
    sample_counts = tables.appearance.groupby("split", observed=True).size().astype(int).to_dict()

    return {
        "schema_version": "segmented-feature-extraction-v1",
        "source_method": tables.method,
        "source_mask": "predicted foreground mask from frozen segmentation artifacts; never reference mask",
        "splits": list(tables.splits),
        "split_sha256": split_sha,
        "feature_sets": {
            "A_seg": list(SEGMENTED_APPEARANCE_FEATURE_NAMES),
            "B": list(MORPHOLOGY_FEATURE_NAMES),
            "C_seg": list(SEGMENTED_FEATURE_NAMES),
        },
        "appearance_definition": {
            "color_space": "CIE Lab via OpenCV RGB float32/255 -> COLOR_RGB2LAB",
            "domain": "pixels where the frozen predicted mask is True",
            "lab_summary": "mean and population standard deviation over mask pixels",
            "glcm_levels": GLCM_LEVELS,
            "glcm_distances": list(GLCM_DISTANCES),
            "glcm_angles_radians": list(GLCM_ANGLES),
            "glcm_quantization": "fixed interval gray // 16 over full 0-255 range",
            "glcm_pair_rule": "both endpoints must lie inside the predicted mask",
            "glcm_symmetric": True,
            "glcm_normalized": True,
            "glcm_aggregation": "mean over the eight distance/angle pairs",
        },
        "morphology_definition": {
            "connectivity": 8,
            "edge_fraction": "foreground pixels in border-touching components / total foreground",
            "area_normalization": "component area / image area",
            "perimeter_normalization": "component perimeter / sqrt(image area)",
            "circularity": "4*pi*area / perimeter^2; zero when perimeter is zero",
            "eccentricity": "skimage.measure.regionprops eccentricity",
            "solidity": "skimage.measure.regionprops solidity",
            "summaries": ["mean", "population_std", "median"],
            "empty_mask_policy": "all numeric features zero, mask_is_empty=1",
        },
        "sample_counts": sample_counts,
        "empty_masks": empty_masks,
        "zero_perimeter_components": zero_perimeter,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
            "scipy": scipy.__version__,
            "scikit_image": skimage.__version__,
        },
    }
