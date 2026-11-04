import numpy as np


def nearest_clusters(lab_pixels, centers, chunk_size=65536):
    pixels = np.asarray(lab_pixels)
    centers = np.asarray(centers, dtype=np.float64)

    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("Expected a pixel table with three Lab columns.")

    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("Expected one or more Lab centers.")

    if not np.isfinite(pixels).all() or not np.isfinite(centers).all():
        raise ValueError("Lab coordinates must be finite.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    labels = np.empty(len(pixels), dtype=np.int32)

    for start in range(0, len(pixels), chunk_size):
        stop = min(start + chunk_size, len(pixels))
        block = pixels[start:stop].astype(np.float64)

        distances = np.empty(
            (len(block), len(centers)), dtype=np.float64
        )

        for cluster_id, center in enumerate(centers):
            difference = block - center
            distances[:, cluster_id] = np.sum(
                difference * difference, axis=1
            )

        labels[start:stop] = np.argmin(distances, axis=1)

    return labels