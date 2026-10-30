import cv2
import numpy as np


def morphology_variants(mask):
    m0 = np.asarray(mask, dtype=bool)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        m0.astype(np.uint8),
        connectivity=8,
    )

    keep = stats[:, cv2.CC_STAT_AREA] >= 16
    keep[0] = False
    m1 = keep[labels]

    kernel = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

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

    return {"M0": m0, "M1": m1, "M2": m2}