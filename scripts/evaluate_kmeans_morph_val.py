"""Thin CLI runner to evaluate Lab K-means (k=2,3,4) M0/M1/M2 variants on validation images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import __version__ as pillow_version

from ebhi_dip.dataset import REFERENCE_THRESHOLD, load_reference_mask, load_rgb, load_split_manifest
from ebhi_dip.metrics import overlap_scores
from ebhi_dip.paths import find_project_root, resolve_manifest_path, verify_frozen_split
from ebhi_dip.provenance import SOURCE_HASH_SCHEME, sha256_bytes, sha256_source_text
from ebhi_dip.segmentation import STAGES, load_kmeans_model, morphology_variants, nearest_clusters, rgb_to_lab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting results/kmeans_morph_val.csv and results/kmeans_morph_val_run.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    output_csv = project_root / "results/kmeans_morph_val.csv"
    run_path = project_root / "results/kmeans_morph_val_run.json"
    if (output_csv.exists() or run_path.exists()) and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing K-means validation results without --overwrite."
        )

    split_sha = verify_frozen_split(project_root)
    val_df = load_split_manifest(project_root, verify=False)
    val_df = val_df.loc[val_df["split"].eq("val")]

    models = {k: load_kmeans_model(project_root, k=k, verify_provenance=True) for k in (2, 3, 4)}
    model_shas = {
        f"k{k}": sha256_bytes(project_root / "models" / f"kmeans_lab_k{k}_mapped.json")
        for k in (2, 3, 4)
    }
    seg_portable_sha = sha256_source_text(project_root / "src/ebhi_dip/segmentation.py")

    results: list[dict[str, object]] = []
    for _, rec in val_df.iterrows():
        img = load_rgb(
            resolve_manifest_path(project_root, rec["image_path"], data_root_override=args.data_root)
        )
        lab = rgb_to_lab(img)
        masks: dict[str, np.ndarray] = {}
        for k, model in models.items():
            labels = nearest_clusters(lab.reshape(-1, 3), model["cluster_centers"]).reshape(img.shape[:2])
            raw_mask = np.isin(labels, model["foreground_clusters"])
            variants = morphology_variants(raw_mask)
            for stage in STAGES:
                masks[f"k{k}_{stage}"] = variants[stage]

        ref = load_reference_mask(
            resolve_manifest_path(project_root, rec["mask_path"], data_root_override=args.data_root),
            threshold=REFERENCE_THRESHOLD,
        )
        masks["all_foreground"] = np.ones_like(ref, dtype=bool)

        for method_name, pred in masks.items():
            dice, iou = overlap_scores(pred, ref)
            results.append(
                {
                    "sample_id": rec["sample_id"],
                    "category": rec["category"],
                    "group_id": rec["group_id"],
                    "split": "val",
                    "method": method_name,
                    "dice": dice,
                    "iou": iou,
                    "predicted_foreground_fraction": float(pred.mean()),
                    "reference_foreground_fraction": float(ref.mean()),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    provenance = {
        "evaluation_subset": "val",
        "image_count": len(val_df),
        "result_row_count": len(results),
        "reference_threshold": REFERENCE_THRESHOLD,
        "aggregation": "Mean image score within each category, then equal-weight mean across categories.",
        "split_sha256": split_sha,
        "model_sha256": model_shas,
        "prediction_helper_sha256": seg_portable_sha,
        "postprocessing_helper_sha256": seg_portable_sha,
        "helper_hash_scheme": SOURCE_HASH_SCHEME,
        "variants": {
            "M0": "raw prediction",
            "M1": "retain components >=16 pixels; connectivity=8",
            "M2": "M1, opening then closing; 3x3 cross; one iteration each; BORDER_REPLICATE",
        },
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
    }
    with run_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(provenance, file, indent=2, allow_nan=False)
        file.write("\n")

    print(f"Evaluated {len(val_df)} validation images | Saved {output_csv} and {run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
