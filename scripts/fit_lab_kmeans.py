"""Thin CLI runner to fit global MiniBatchKMeans (k=2,3,4) on sampled training Lab pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn
from sklearn.cluster import MiniBatchKMeans

from ebhi_dip.paths import PIXEL_CACHE_REL, find_project_root, verify_frozen_split
from ebhi_dip.provenance import sha256_bytes

SHARED_PARAMETERS = {
    "init": "k-means++",
    "batch_size": 4096,
    "init_size": 12288,
    "n_init": 10,
    "random_state": 42,
    "max_iter": 100,
    "max_no_improvement": 20,
    "tol": 0.0,
    "reassignment_ratio": 0.01,
    "compute_labels": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing models/kmeans_lab_k{2,3,4}.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    model_dir = project_root / "models"
    targets = [model_dir / f"kmeans_lab_k{k}.json" for k in (2, 3, 4)]
    if any(p.exists() for p in targets) and not args.overwrite:
        raise FileExistsError("Refusing to overwrite existing fitted K-means JSON models without --overwrite.")

    split_sha = verify_frozen_split(project_root)
    cache_path = project_root / PIXEL_CACHE_REL
    cache_sha = sha256_bytes(cache_path)

    with np.load(cache_path, allow_pickle=False) as cache:
        x_lab = np.ascontiguousarray(cache["lab_pixels"])
        if cache["split_sha256"].item() != split_sha or cache["subset"].item() != "train":
            raise ValueError("Sampled pixel cache provenance does not match frozen training split.")
        sampling = {
            "training_images": len(cache["sample_ids"]),
            "pixels_per_image": int(cache["pixels_per_image"].item()),
            "seed": int(cache["seed"].item()),
            "opencv_version": cache["opencv_version"].item(),
            "numpy_version": cache["numpy_version"].item(),
        }

    if x_lab.ndim != 2 or x_lab.shape[1] != 3 or x_lab.dtype != np.float32 or not np.isfinite(x_lab).all():
        raise ValueError("Unexpected sampled Lab array format.")

    model_dir.mkdir(parents=True, exist_ok=True)
    for k in (2, 3, 4):
        model = MiniBatchKMeans(n_clusters=k, **SHARED_PARAMETERS)
        model.fit(x_lab)
        centers = model.cluster_centers_
        counts = np.bincount(model.labels_, minlength=k)

        artifact = {
            "method": "lab_kmeans",
            "estimator": "sklearn.cluster.MiniBatchKMeans",
            "fit_subset": "train",
            "color_conversion": "RGB float32 in [0,1] -> COLOR_RGB2LAB",
            "channel_order": ["L*", "a*", "b*"],
            "feature_scaling": "none",
            "assignment_rule": "nearest center by squared Euclidean distance",
            "cluster_centers": centers.tolist(),
            "cluster_centers_dtype": str(centers.dtype),
            "foreground_clusters": None,
            "fit_parameters": model.get_params(deep=False),
            "training_pixel_count": len(x_lab),
            "sampling": sampling,
            "split_sha256": split_sha,
            "sample_cache_sha256": cache_sha,
            "versions": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "scipy": scipy.__version__,
                "numpy": np.__version__,
            },
            "fit_diagnostics": {
                "inertia": float(model.inertia_),
                "mean_squared_lab_distance": float(model.inertia_) / len(x_lab),
                "n_iter": int(model.n_iter_),
                "n_steps": int(model.n_steps_),
                "assigned_pixel_counts": counts.tolist(),
            },
        }
        out_path = model_dir / f"kmeans_lab_k{k}.json"
        with out_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(artifact, file, indent=2, allow_nan=False)
            file.write("\n")
        print(f"Saved fitted model k={k}: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
