"""Project root discovery, frozen split verification, and data-path resolution."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

FROZEN_SPLIT_REL = Path("data/metadata/split_manifest.csv")
FROZEN_SPLIT_SHA_REL = Path("data/metadata/split_manifest.sha256")
RAW_DATASET_REL = Path("data/raw/EBHI-SEG")
PIXEL_CACHE_REL = Path("data/processed/kmeans_train_pixels.npz")

DATA_ROOT_ENV_VAR = "EBHI_DIP_DATA_ROOT"


def _has_root_markers(candidate: Path) -> bool:
    return (
        candidate.is_dir()
        and (candidate / FROZEN_SPLIT_REL).is_file()
        and (candidate / FROZEN_SPLIT_SHA_REL).is_file()
    )


def find_project_root(
    start: str | Path | None = None,
    override: str | Path | None = None,
) -> Path:
    """Locate the EBHI-DIP project root via explicit override or ancestor markers."""
    if override is not None:
        candidate = Path(override).expanduser().resolve()
        if not _has_root_markers(candidate):
            raise FileNotFoundError(
                f"Invalid project root override '{candidate}': missing "
                f"'{FROZEN_SPLIT_REL.as_posix()}' or '{FROZEN_SPLIT_SHA_REL.as_posix()}'."
            )
        return candidate

    base = Path(start).expanduser().resolve() if start is not None else Path.cwd().resolve()
    if base.is_file():
        base = base.parent

    for candidate in (base, *base.parents):
        if _has_root_markers(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not discover EBHI-DIP project root from '{base}'. "
        f"Expected markers '{FROZEN_SPLIT_REL.as_posix()}' and '{FROZEN_SPLIT_SHA_REL.as_posix()}'."
    )


def verify_frozen_split(project_root: str | Path) -> str:
    """Verify exact raw bytes of split_manifest.csv against split_manifest.sha256."""
    root = Path(project_root).expanduser().resolve()
    split_path = root / FROZEN_SPLIT_REL
    sha_path = root / FROZEN_SPLIT_SHA_REL

    if not split_path.is_file():
        raise FileNotFoundError(f"Missing frozen split file: {split_path}")
    if not sha_path.is_file():
        raise FileNotFoundError(f"Missing frozen split checksum file: {sha_path}")

    tokens = sha_path.read_text(encoding="utf-8").split()
    if not tokens:
        raise ValueError(f"Empty frozen split checksum sidecar: {sha_path}")

    expected_sha = tokens[0].strip().lower()
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        raise ValueError(f"Invalid SHA-256 digest in sidecar '{sha_path}': {tokens[0]!r}")

    actual_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            "Frozen split raw-byte checksum mismatch for "
            f"'{split_path}': expected {expected_sha}, got {actual_sha}."
        )

    return actual_sha


def resolve_data_root(
    project_root: str | Path,
    data_root_override: str | Path | None = None,
) -> Path:
    """Resolve the EBHI-SEG raw dataset directory without mutating manifest paths."""
    root = Path(project_root).expanduser().resolve()
    raw_override = data_root_override
    if raw_override is None:
        env_value = os.environ.get(DATA_ROOT_ENV_VAR, "").strip()
        if env_value:
            raw_override = env_value

    if raw_override is None:
        return root / RAW_DATASET_REL

    override_path = Path(raw_override).expanduser().resolve()
    nested = override_path / RAW_DATASET_REL
    if nested.is_dir():
        return nested
    return override_path


def resolve_manifest_path(
    project_root: str | Path,
    relative_path: str | Path,
    data_root_override: str | Path | None = None,
) -> Path:
    """Resolve a manifest-recorded relative path against project root or data-root override."""
    root = Path(project_root).expanduser().resolve()
    rel = Path(relative_path)
    if rel.is_absolute():
        raise ValueError(f"Manifest path must be relative to project root, got: {relative_path}")

    raw_override = data_root_override
    if raw_override is None:
        env_value = os.environ.get(DATA_ROOT_ENV_VAR, "").strip()
        if env_value:
            raw_override = env_value

    if raw_override is None:
        return root / rel

    override_path = Path(raw_override).expanduser().resolve()
    direct_candidate = override_path / rel
    if direct_candidate.exists():
        return direct_candidate

    rel_posix = rel.as_posix()
    prefix = RAW_DATASET_REL.as_posix() + "/"
    if rel_posix.startswith(prefix):
        sub_rel = Path(rel_posix[len(prefix):])
        dataset_dir = resolve_data_root(root, data_root_override=override_path)
        return dataset_dir / sub_rel

    return direct_candidate


def has_raw_dataset(
    project_root: str | Path,
    data_root_override: str | Path | None = None,
) -> bool:
    """Return True if the EBHI-SEG raw dataset directory exists and contains category folders."""
    dataset_dir = resolve_data_root(project_root, data_root_override=data_root_override)
    if not dataset_dir.is_dir():
        return False
    return any((dataset_dir / child.name / "image").is_dir() for child in dataset_dir.iterdir() if child.is_dir())
