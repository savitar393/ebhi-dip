"""Dataset constants, image/mask loaders, and frozen cohort metadata access."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ebhi_dip.paths import FROZEN_SPLIT_REL, verify_frozen_split

EXPECTED_CATEGORIES: tuple[str, ...] = (
    "Adenocarcinoma",
    "High-grade IN",
    "Low-grade IN",
    "Normal",
    "Polyp",
    "Serrated adenoma",
)
IMAGE_SHAPE_RGB: tuple[int, int, int] = (224, 224, 3)
MASK_SHAPE_L: tuple[int, int] = (224, 224)
REFERENCE_THRESHOLD: int = 128
ALLOWED_SPLITS: tuple[str, ...] = ("train", "val", "test")


def canonical_group_id(image_path_or_stem: str | Path) -> str:
    """Extract provisional uppercase filename-prefix group ID before the first '-'."""
    stem = Path(image_path_or_stem).stem
    prefix = stem.split("-")[0].strip().upper()
    if not prefix:
        raise ValueError(f"Could not extract filename prefix from '{image_path_or_stem}'.")
    return prefix


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as a contiguous uint8 RGB array of shape (H, W, 3)."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing RGB image file: {image_path}")
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected 3-channel RGB image at '{image_path}', got {rgb.shape}.")
    return np.ascontiguousarray(rgb)


def load_raw_mask(path: str | Path) -> np.ndarray:
    """Load a raw single-channel uint8 mask array of shape (H, W) for encoding inspection."""
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing mask file: {mask_path}")
    with Image.open(mask_path) as source:
        raw = np.asarray(
            source if source.mode == "L" else source.convert("L"),
            dtype=np.uint8,
        )
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D grayscale mask at '{mask_path}', got {raw.shape}.")
    return np.ascontiguousarray(raw)


def load_reference_mask(
    path: str | Path,
    threshold: int = REFERENCE_THRESHOLD,
) -> np.ndarray:
    """Load binary reference mask using raw_mask >= threshold (default 128)."""
    if not (0 <= int(threshold) <= 255):
        raise ValueError(f"Reference threshold must be in [0, 255], got {threshold}.")
    return load_raw_mask(path) >= int(threshold)


def compute_pixel_sha256(image_rgb: np.ndarray) -> str:
    """Compute the decoded RGB pixel SHA-256 fingerprint matching historical audit_duplicates."""
    rgb = np.asarray(image_rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected uint8 RGB image (H, W, 3), got dtype={rgb.dtype}, shape={rgb.shape}.")
    height, width, _ = rgb.shape
    digest = hashlib.sha256()
    digest.update(f"RGB:{width}:{height}:".encode("ascii"))
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def load_split_manifest(
    project_root: str | Path,
    verify: bool = True,
) -> pd.DataFrame:
    """Load and validate data/metadata/split_manifest.csv without modifying disk bytes."""
    root = Path(project_root).expanduser().resolve()
    if verify:
        verify_frozen_split(root)

    split_path = root / FROZEN_SPLIT_REL
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing frozen split CSV: {split_path}")

    split_df = pd.read_csv(split_path, encoding="utf-8-sig", keep_default_na=False)
    required_cols = {
        "sample_id",
        "category",
        "image_path",
        "mask_path",
        "pair_status",
        "pixel_sha256",
        "group_id",
        "split",
    }
    missing = required_cols - set(split_df.columns)
    if missing:
        raise ValueError(f"Frozen split CSV missing columns: {sorted(missing)}")

    if not split_df["sample_id"].is_unique:
        raise ValueError("Frozen split CSV contains duplicate sample_id values.")
    if not split_df["split"].isin(ALLOWED_SPLITS).all():
        raise ValueError("Frozen split CSV contains unknown split labels.")
    if split_df.groupby("group_id", observed=True)["split"].nunique().gt(1).any():
        raise ValueError("A filename group crosses train/val/test subsets.")
    if split_df.groupby("pixel_sha256", observed=True)["split"].nunique().gt(1).any():
        raise ValueError("A recorded pixel fingerprint crosses train/val/test subsets.")

    return split_df
