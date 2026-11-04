import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.cluster import MiniBatchKMeans


project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"
split_path = metadata_dir / "split_manifest.csv"
cache_path = (
    project_root / "data" / "processed" / "kmeans_train_pixels.npz"
)

expected_checksum = (
    (metadata_dir / "split_manifest.sha256")
    .read_text(encoding="utf-8")
    .split()[0]
)

if hashlib.sha256(split_path.read_bytes()).hexdigest() != expected_checksum:
    raise SystemExit("Frozen split checksum mismatch.")

cache_checksum = hashlib.sha256(cache_path.read_bytes()).hexdigest()

with np.load(cache_path, allow_pickle=False) as cache:
    X = cache["lab_pixels"]

    if cache["split_sha256"].item() != expected_checksum:
        raise SystemExit("Pixel cache belongs to a different split.")

    if cache["subset"].item() != "train":
        raise SystemExit("Expected a training-only pixel cache.")

    sampling = {
        "training_images": len(cache["sample_ids"]),
        "pixels_per_image": int(cache["pixels_per_image"].item()),
        "seed": int(cache["seed"].item()),
        "opencv_version": cache["opencv_version"].item(),
        "numpy_version": cache["numpy_version"].item(),
    }

if (
    X.ndim != 2
    or X.shape[1] != 3
    or X.dtype != np.float32
    or not np.isfinite(X).all()
):
    raise SystemExit("Unexpected Lab sample format.")

X = np.ascontiguousarray(X)

model_dir = project_root / "models"
model_dir.mkdir(parents=True, exist_ok=True)

shared_parameters = {
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

print("scikit-learn:", sklearn.__version__)
print("Training pixel table:", X.shape)
print("Training images:", sampling["training_images"])

for k in (2, 3, 4):
    print(f"\nFitting k={k}...", flush=True)

    model = MiniBatchKMeans(
        n_clusters=k,
        **shared_parameters,
    )

    # Fit using Lab coordinates only.
    model.fit(X)

    centers = model.cluster_centers_
    counts = np.bincount(model.labels_, minlength=k)

    if centers.shape != (k, 3) or not np.isfinite(centers).all():
        raise SystemExit(f"Unexpected fitted centers for k={k}.")

    mean_squared_distance = float(model.inertia_) / len(X)

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
        "training_pixel_count": len(X),
        "sampling": sampling,
        "split_sha256": expected_checksum,
        "sample_cache_sha256": cache_checksum,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        },
        "fit_diagnostics": {
            "inertia": float(model.inertia_),
            "mean_squared_lab_distance": mean_squared_distance,
            "n_iter": int(model.n_iter_),
            "n_steps": int(model.n_steps_),
            "assigned_pixel_counts": counts.tolist(),
        },
    }

    output_path = model_dir / f"kmeans_lab_k{k}.json"

    with output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(artifact, file, indent=2, allow_nan=False)
        file.write("\n")

    print("Mini-batches processed:", model.n_steps_)
    print(f"Mean squared Lab distance: {mean_squared_distance:.4f}")
    print(f"{'Cluster':>7} {'Pixels':>10} {'L*':>10} {'a*':>10} {'b*':>10}")

    for cluster_id, (center, count) in enumerate(zip(centers, counts)):
        print(
            f"{cluster_id:>7} {int(count):>10} "
            f"{center[0]:>10.4f} "
            f"{center[1]:>10.4f} "
            f"{center[2]:>10.4f}"
        )

    print("Saved:", output_path)