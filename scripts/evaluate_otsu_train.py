"""Thin CLI runner to evaluate dark/bright Otsu polarity on the training split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ebhi_dip.dataset import EXPECTED_CATEGORIES, REFERENCE_THRESHOLD, load_reference_mask, load_rgb, load_split_manifest
from ebhi_dip.metrics import category_macro_scores, overlap_scores
from ebhi_dip.paths import find_project_root, resolve_manifest_path, verify_frozen_split
from ebhi_dip.segmentation import otsu_threshold_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting results/otsu_train.csv and configs/otsu_raw.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    output_csv = project_root / "results/otsu_train.csv"
    config_path = project_root / "configs/otsu_raw.json"
    if (output_csv.exists() or config_path.exists()) and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing results/otsu_train.csv or configs/otsu_raw.json without --overwrite."
        )

    split_sha = verify_frozen_split(project_root)
    split_df = load_split_manifest(project_root, verify=False)
    train_df = split_df.loc[split_df["split"].eq("train")]

    rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for _, rec in train_df.iterrows():
        img = load_rgb(
            resolve_manifest_path(project_root, rec["image_path"], data_root_override=args.data_root)
        )
        ref = load_reference_mask(
            resolve_manifest_path(project_root, rec["mask_path"], data_root_override=args.data_root),
            threshold=REFERENCE_THRESHOLD,
        )
        thr, dark_mask = otsu_threshold_mask(img, foreground_polarity="dark")
        bright_mask = ~dark_mask
        control_mask = np.ones_like(dark_mask, dtype=bool)

        entry: dict[str, object] = {
            "sample_id": rec["sample_id"],
            "category": rec["category"],
            "group_id": rec["group_id"],
            "otsu_threshold": thr,
            "reference_threshold": REFERENCE_THRESHOLD,
        }
        for name, pred in (("dark", dark_mask), ("bright", bright_mask), ("all_foreground", control_mask)):
            dice, iou = overlap_scores(pred, ref)
            entry[f"{name}_dice"] = dice
            entry[f"{name}_iou"] = iou
            long_rows.append(
                {
                    "sample_id": rec["sample_id"],
                    "category": rec["category"],
                    "method": name,
                    "dice": dice,
                    "iou": iou,
                }
            )
        rows.append(entry)

    macro = category_macro_scores(pd.DataFrame(long_rows), expected_categories=EXPECTED_CATEGORIES)
    chosen_polarity = "dark" if macro.loc["dark", "mean_category_dice"] >= macro.loc["bright", "mean_category_dice"] else "bright"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    config = {
        "method": "otsu",
        "grayscale_conversion": "RGB2GRAY",
        "threshold_mode": "per_image_otsu",
        "foreground_polarity": chosen_polarity,
        "postprocessing": "none",
        "reference_threshold": REFERENCE_THRESHOLD,
        "polarity_selected_on": "train",
        "selection_metric": "mean_category_dice",
        "training_mean_category_dice": float(macro.loc[chosen_polarity, "mean_category_dice"]),
        "split_sha256": split_sha,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(config, file, indent=2)
        file.write("\n")

    print(f"Chosen Otsu polarity: {chosen_polarity} | Saved {output_csv} and {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
