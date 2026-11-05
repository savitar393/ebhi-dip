"""Thin CLI runner to map K-means Lab clusters to foreground using sampled training annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ebhi_dip.dataset import REFERENCE_THRESHOLD
from ebhi_dip.paths import PIXEL_CACHE_REL, find_project_root, verify_frozen_split
from ebhi_dip.provenance import SOURCE_HASH_SCHEME, sha256_bytes, sha256_source_text
from ebhi_dip.segmentation import nearest_clusters

FOREGROUND_RATE_THRESHOLD = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing models/kmeans_lab_k{2,3,4}_mapped.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    model_dir = project_root / "models"
    targets = [model_dir / f"kmeans_lab_k{k}_mapped.json" for k in (2, 3, 4)]
    if any(p.exists() for p in targets) and not args.overwrite:
        raise FileExistsError("Refusing to overwrite existing mapped K-means JSON models without --overwrite.")

    split_sha = verify_frozen_split(project_root)
    cache_path = project_root / PIXEL_CACHE_REL
    cache_sha = sha256_bytes(cache_path)
    seg_portable_sha = sha256_source_text(project_root / "src/ebhi_dip/segmentation.py")

    with np.load(cache_path, allow_pickle=False) as cache:
        x_lab = cache["lab_pixels"]
        y_ref = cache["reference_foreground"]
        ref_thr = int(cache["reference_threshold"].item())
        if cache["subset"].item() != "train" or cache["split_sha256"].item() != split_sha:
            raise ValueError("Expected sampled training pixel cache matching frozen split.")

    if y_ref.shape != (len(x_lab),) or y_ref.dtype != np.bool_ or ref_thr != REFERENCE_THRESHOLD:
        raise ValueError("Unexpected sampled reference format.")

    for k in (2, 3, 4):
        model_path = model_dir / f"kmeans_lab_k{k}.json"
        source_sha = sha256_bytes(model_path)
        artifact = json.loads(model_path.read_text(encoding="utf-8"))
        if (
            artifact["fit_subset"] != "train"
            or artifact["split_sha256"] != split_sha
            or artifact["sample_cache_sha256"] != cache_sha
        ):
            raise ValueError(f"Model/cache provenance mismatch for k={k}.")

        labels = nearest_clusters(x_lab, artifact["cluster_centers"])
        counts = np.bincount(labels, minlength=k)
        if np.any(counts == 0):
            raise RuntimeError(f"k={k}: empty cluster encountered.")

        fg_counts = np.bincount(labels[y_ref], minlength=k)
        rates = fg_counts / counts
        is_fg = rates >= FOREGROUND_RATE_THRESHOLD
        fg_clusters = np.flatnonzero(is_fg).tolist()

        artifact["foreground_clusters"] = fg_clusters
        artifact["foreground_mapping"] = {
            "learned_on": "sampled_train_pixels",
            "rule": "reference_foreground_rate >= threshold",
            "threshold": FOREGROUND_RATE_THRESHOLD,
            "reference_threshold": ref_thr,
            "cluster_pixel_counts": counts.tolist(),
            "cluster_reference_foreground_counts": fg_counts.tolist(),
            "cluster_reference_foreground_rates": rates.tolist(),
            "numpy_version": np.__version__,
        }
        artifact["prediction"] = {
            "function": "ebhi_dip.segmentation.nearest_clusters",
            "distance_dtype": "float64",
            "tie_break": "lowest_cluster_id",
            "helper_sha256": seg_portable_sha,
            "helper_hash_scheme": SOURCE_HASH_SCHEME,
        }
        artifact["source_fit_sha256"] = source_sha

        out_path = model_dir / f"kmeans_lab_k{k}_mapped.json"
        with out_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(artifact, file, indent=2, allow_nan=False)
            file.write("\n")
        print(f"k={k}: foreground_clusters={fg_clusters} | Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
