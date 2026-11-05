"""Thin CLI runner to sample 1,000 distinct Lab pixels per training image (seed=42)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ebhi_dip.dataset import IMAGE_SHAPE_RGB, MASK_SHAPE_L, REFERENCE_THRESHOLD, load_raw_mask, load_rgb, load_split_manifest
from ebhi_dip.paths import PIXEL_CACHE_REL, find_project_root, resolve_manifest_path, verify_frozen_split
from ebhi_dip.segmentation import rgb_to_lab

SEED = 42
PIXELS_PER_IMAGE = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing data/processed/kmeans_train_pixels.npz.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    output_path = project_root / PIXEL_CACHE_REL
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing '{output_path}' without --overwrite."
        )

    split_sha = verify_frozen_split(project_root)
    split_df = load_split_manifest(project_root, verify=False)
    train_df = split_df.loc[split_df["split"].eq("train")].sort_values("sample_id", ascending=True)

    rng = np.random.default_rng(SEED)
    lab_parts: list[np.ndarray] = []
    ref_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []

    for _, rec in train_df.iterrows():
        img = load_rgb(
            resolve_manifest_path(project_root, rec["image_path"], data_root_override=args.data_root)
        )
        mask = load_raw_mask(
            resolve_manifest_path(project_root, rec["mask_path"], data_root_override=args.data_root)
        )
        if img.shape != IMAGE_SHAPE_RGB or mask.shape != MASK_SHAPE_L:
            raise ValueError(f"Unexpected dimensions for {rec['sample_id']}.")

        lab = rgb_to_lab(img).reshape(-1, 3)
        ref = mask.reshape(-1) >= REFERENCE_THRESHOLD
        indices = rng.choice(lab.shape[0], size=PIXELS_PER_IMAGE, replace=False)

        lab_parts.append(lab[indices])
        ref_parts.append(ref[indices])
        index_parts.append(indices.astype(np.uint32))

    x_lab = np.concatenate(lab_parts, axis=0)
    y_ref = np.concatenate(ref_parts, axis=0)
    if not np.isfinite(x_lab).all():
        raise ValueError("Sampled Lab coordinates contain non-finite values.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        lab_pixels=x_lab,
        reference_foreground=y_ref,
        sample_ids=train_df["sample_id"].to_numpy(),
        sampled_pixel_indices=np.stack(index_parts),
        pixels_per_image=PIXELS_PER_IMAGE,
        reference_threshold=REFERENCE_THRESHOLD,
        seed=SEED,
        subset="train",
        sample_order="sample_id ascending",
        color_conversion="RGB float32 / 255 -> COLOR_RGB2LAB",
        split_sha256=split_sha,
        opencv_version=cv2.__version__,
        numpy_version=np.__version__,
    )
    print(f"Sampled {len(x_lab)} pixels from {len(train_df)} images | Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
