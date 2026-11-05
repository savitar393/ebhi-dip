"""Classical segmentation numerical core: color spaces, Otsu, Lab K-means, and morphology."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ebhi_dip.dataset import REFERENCE_THRESHOLD
from ebhi_dip.paths import verify_frozen_split
from ebhi_dip.provenance import load_provenance_migration, verify_helper_provenance

STAGES: tuple[str, ...] = ("M0", "M1", "M2")
SUPPORTED_METHODS: tuple[str, ...] = (
    "otsu_M0",
    "otsu_M1",
    "otsu_M2",
    "k2_M0",
    "k2_M1",
    "k2_M2",
    "k3_M0",
    "k3_M1",
    "k3_M2",
    "k4_M0",
    "k4_M1",
    "k4_M2",
    "all_foreground",
)

OTSU_CONFIG_REL = Path("configs/otsu_raw.json")
MORPH_RUN_REL = Path("results/kmeans_morph_val_run.json")


def _validate_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image_rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"Expected uint8 RGB image array of shape (H, W, 3), got dtype={rgb.dtype}, shape={rgb.shape}."
        )
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("RGB image dimensions must be non-zero.")
    return np.ascontiguousarray(rgb)


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to OpenCV uint8 grayscale (COLOR_RGB2GRAY)."""
    rgb = _validate_rgb_image(image_rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image to physical float32 CIE Lab via float32/255 -> COLOR_RGB2LAB."""
    rgb = _validate_rgb_image(image_rgb)
    rgb_float = rgb.astype(np.float32) / np.float32(255.0)
    return cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)


def otsu_threshold_mask(
    image_rgb: np.ndarray,
    foreground_polarity: str = "dark",
) -> tuple[float, np.ndarray]:
    """Compute per-image Otsu threshold and raw binary foreground mask with fixed polarity."""
    if foreground_polarity not in ("dark", "bright"):
        raise ValueError(
            f"Unsupported Otsu foreground polarity '{foreground_polarity}'; expected 'dark' or 'bright'."
        )
    gray = rgb_to_gray(image_rgb)
    threshold, bright_u8 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    bright = bright_u8 > 0
    mask = ~bright if foreground_polarity == "dark" else bright
    return float(threshold), mask


def nearest_clusters(
    lab_pixels: np.ndarray,
    centers: np.ndarray | list,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Assign each Lab pixel to its nearest cluster center using float64 squared distance.

    Exact distance ties resolve deterministically to the lowest cluster index via np.argmin.
    """
    pixels = np.asarray(lab_pixels)
    centers_arr = np.asarray(centers, dtype=np.float64)

    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("Expected a pixel table with three Lab columns.")
    if centers_arr.ndim != 2 or centers_arr.shape[1] != 3 or len(centers_arr) == 0:
        raise ValueError("Expected one or more Lab centers.")
    if not np.isfinite(pixels).all() or not np.isfinite(centers_arr).all():
        raise ValueError("Lab coordinates must be finite.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    labels = np.empty(len(pixels), dtype=np.int32)
    for start in range(0, len(pixels), chunk_size):
        stop = min(start + chunk_size, len(pixels))
        block = pixels[start:stop].astype(np.float64)
        distances = np.empty((len(block), len(centers_arr)), dtype=np.float64)
        for cluster_id, center in enumerate(centers_arr):
            difference = block - center
            distances[:, cluster_id] = np.sum(difference * difference, axis=1)
        labels[start:stop] = np.argmin(distances, axis=1)

    return labels


def morphology_variants(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Compute M0 (raw), M1 (8-connected area >= 16), and M2 (M1 + 3x3 cross open then close)."""
    m0 = np.asarray(mask)
    if m0.ndim != 2:
        raise ValueError(f"Expected 2D binary mask for morphology, got shape {m0.shape}.")
    m0_bool = m0.astype(bool, copy=False)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        m0_bool.astype(np.uint8),
        connectivity=8,
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= 16
    keep[0] = False
    m1 = keep[labels]

    kernel = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    opened = cv2.morphologyEx(
        m1.astype(np.uint8),
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_REPLICATE,
    )
    m2 = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_REPLICATE,
    ).astype(bool)

    return {"M0": m0_bool.copy(), "M1": m1, "M2": m2}


def load_otsu_config(project_root: str | Path) -> dict[str, object]:
    """Load and validate train-selected Otsu configuration against the frozen split."""
    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)

    config_path = root / OTSU_CONFIG_REL
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Otsu configuration: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "method": "otsu",
        "grayscale_conversion": "RGB2GRAY",
        "threshold_mode": "per_image_otsu",
        "postprocessing": "none",
        "polarity_selected_on": "train",
        "reference_threshold": REFERENCE_THRESHOLD,
        "split_sha256": split_sha,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Invalid Otsu configuration field '{key}': expected {value!r}, got {config.get(key)!r}."
            )
    if config.get("foreground_polarity") not in ("dark", "bright"):
        raise ValueError(f"Unknown Otsu foreground polarity: {config.get('foreground_polarity')!r}")
    return config


def load_kmeans_model(
    project_root: str | Path,
    k: int,
    verify_provenance: bool = True,
) -> dict[str, object]:
    """Load and validate a train-fitted mapped Lab K-means artifact (k in {2, 3, 4})."""
    if k not in (2, 3, 4):
        raise ValueError(f"Unsupported K-means cluster count k={k}; expected 2, 3, or 4.")

    root = Path(project_root).expanduser().resolve()
    split_sha = verify_frozen_split(root)
    model_path = root / "models" / f"kmeans_lab_k{k}_mapped.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing mapped K-means artifact: {model_path}")

    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("fit_subset") != "train" or model.get("split_sha256") != split_sha:
        raise ValueError(f"Split/subset provenance mismatch in '{model_path.name}'.")

    mapping = model.get("foreground_mapping", {})
    if (
        mapping.get("learned_on") != "sampled_train_pixels"
        or mapping.get("threshold") != 0.5
        or mapping.get("reference_threshold") != REFERENCE_THRESHOLD
    ):
        raise ValueError(f"Foreground mapping contract mismatch in '{model_path.name}'.")

    centers = model.get("cluster_centers", [])
    if len(centers) != k:
        raise ValueError(f"Expected {k} cluster centers in '{model_path.name}', got {len(centers)}.")

    if verify_provenance:
        helper_sha = str(model["prediction"]["helper_sha256"])
        verify_helper_provenance(root, "kmeans_prediction", helper_sha)

    return model


def _verify_postprocessing_provenance(project_root: Path) -> None:
    run_path = project_root / MORPH_RUN_REL
    if run_path.is_file():
        run_meta = json.loads(run_path.read_text(encoding="utf-8"))
        recorded = str(run_meta["postprocessing_helper_sha256"])
    else:
        migration = load_provenance_migration(project_root)
        recorded = str(migration["legacy_helpers"]["postprocessing"]["recorded_crlf_sha256"])
    verify_helper_provenance(project_root, "postprocessing", recorded)


def predict_mask(
    image_rgb: np.ndarray,
    method_id: str,
    project_root: str | Path,
) -> np.ndarray:
    """Predict a fixed HxW boolean foreground mask from image_rgb and saved artifacts only.

    Never reads or accepts a per-image reference mask, and never fits or selects parameters.
    """
    rgb = _validate_rgb_image(image_rgb)
    if method_id not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported segmentation method '{method_id}'. Supported: {SUPPORTED_METHODS}."
        )

    root = Path(project_root).expanduser().resolve()
    if method_id == "all_foreground":
        verify_frozen_split(root)
        return np.ones(rgb.shape[:2], dtype=bool)

    family, stage = method_id.split("_", 1)
    if family == "otsu":
        config = load_otsu_config(root)
        _, raw_mask = otsu_threshold_mask(rgb, str(config["foreground_polarity"]))
    else:
        k = int(family[1:])
        model = load_kmeans_model(root, k=k, verify_provenance=True)
        lab = rgb_to_lab(rgb)
        labels = nearest_clusters(lab.reshape(-1, 3), model["cluster_centers"]).reshape(
            rgb.shape[:2]
        )
        raw_mask = np.isin(labels, model["foreground_clusters"])

    if stage == "M0":
        return raw_mask

    _verify_postprocessing_provenance(root)
    return morphology_variants(raw_mask)[stage]
