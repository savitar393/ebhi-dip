import hashlib
import json
from pathlib import Path

import numpy as np

from kmeans_prediction import nearest_clusters


FOREGROUND_RATE_THRESHOLD = 0.5

project_root = Path(__file__).resolve().parent
metadata_dir = project_root / "data" / "metadata"
model_dir = project_root / "models"

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

helper_checksum = hashlib.sha256(
    (project_root / "kmeans_prediction.py").read_bytes()
).hexdigest()

with np.load(cache_path, allow_pickle=False) as cache:
    X = cache["lab_pixels"]
    y = cache["reference_foreground"]
    reference_threshold = int(cache["reference_threshold"].item())

    if (
        cache["subset"].item() != "train"
        or cache["split_sha256"].item() != expected_checksum
    ):
        raise SystemExit("Expected a cache from the frozen training split.")

if (
    y.shape != (len(X),)
    or y.dtype != np.bool_
    or reference_threshold != 128
):
    raise SystemExit("Unexpected sampled reference format.")

print("Training pixels used for mapping:", len(X))
print(f"Sampled reference foreground: {100 * y.mean():.2f}%")
print("Foreground-rate threshold:", FOREGROUND_RATE_THRESHOLD)

for k in (2, 3, 4):
    model_path = model_dir / f"kmeans_lab_k{k}.json"
    model_bytes = model_path.read_bytes()
    artifact = json.loads(model_bytes)

    if (
        artifact["fit_subset"] != "train"
        or artifact["split_sha256"] != expected_checksum
        or artifact["sample_cache_sha256"] != cache_checksum
    ):
        raise SystemExit(f"Model/cache provenance mismatch for k={k}.")

    centers = artifact["cluster_centers"]

    if len(centers) != k:
        raise SystemExit(f"Unexpected center count for k={k}.")

    labels = nearest_clusters(X, centers)

    counts = np.bincount(labels, minlength=k)

    if np.any(counts == 0):
        raise SystemExit(
            f"k={k}: a cluster has no assigned pixels. "
            "Send this output before proceeding."
        )

    foreground_counts = np.bincount(labels[y], minlength=k)
    rates = foreground_counts / counts

    is_foreground = rates >= FOREGROUND_RATE_THRESHOLD
    foreground_clusters = np.flatnonzero(is_foreground).tolist()

    artifact["foreground_clusters"] = foreground_clusters

    artifact["foreground_mapping"] = {
        "learned_on": "sampled_train_pixels",
        "rule": "reference_foreground_rate >= threshold",
        "threshold": FOREGROUND_RATE_THRESHOLD,
        "reference_threshold": reference_threshold,
        "cluster_pixel_counts": counts.tolist(),
        "cluster_reference_foreground_counts": foreground_counts.tolist(),
        "cluster_reference_foreground_rates": rates.tolist(),
        "numpy_version": np.__version__,
    }

    artifact["prediction"] = {
        "function": "kmeans_prediction.nearest_clusters",
        "distance_dtype": "float64",
        "tie_break": "lowest_cluster_id",
        "helper_sha256": helper_checksum,
    }

    artifact["source_fit_sha256"] = hashlib.sha256(
        model_bytes
    ).hexdigest()

    output_path = model_dir / f"kmeans_lab_k{k}_mapped.json"

    with output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as file:
        json.dump(artifact, file, indent=2, allow_nan=False)
        file.write("\n")

    print(f"\nk={k}")
    print(
        f"{'Cluster':>7} {'Pixels':>10} {'Ref FG':>10} "
        f"{'FG rate':>10} {'Mapping':>12}"
    )

    for cluster_id in range(k):
        meaning = (
            "foreground" if is_foreground[cluster_id]
            else "background"
        )

        print(
            f"{cluster_id:>7} "
            f"{int(counts[cluster_id]):>10} "
            f"{int(foreground_counts[cluster_id]):>10} "
            f"{rates[cluster_id]:>10.4f} "
            f"{meaning:>12}"
        )

    coverage = counts[is_foreground].sum() / len(X)

    print("Foreground clusters:", foreground_clusters)
    print(f"Sampled predicted foreground: {100 * coverage:.2f}%")
    print("Saved:", output_path)