"""Thin CLI runner to evaluate raw train-selected Otsu on the validation split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ebhi_dip.dataset import load_reference_mask, load_rgb, load_split_manifest
from ebhi_dip.metrics import overlap_scores
from ebhi_dip.paths import find_project_root, resolve_manifest_path
from ebhi_dip.segmentation import load_otsu_config, otsu_threshold_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Explicit repository root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional EBHI-SEG data root override.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing results/otsu_val.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = find_project_root(start=Path(__file__), override=args.project_root)
    output_path = project_root / "results/otsu_val.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing '{output_path}' without --overwrite."
        )

    config = load_otsu_config(project_root)
    polarity = str(config["foreground_polarity"])
    ref_thr = int(config["reference_threshold"])
    val_df = load_split_manifest(project_root, verify=False)
    val_df = val_df.loc[val_df["split"].eq("val")]

    rows: list[dict[str, object]] = []
    for _, rec in val_df.iterrows():
        img = load_rgb(
            resolve_manifest_path(project_root, rec["image_path"], data_root_override=args.data_root)
        )
        ref = load_reference_mask(
            resolve_manifest_path(project_root, rec["mask_path"], data_root_override=args.data_root),
            threshold=ref_thr,
        )
        thr, pred = otsu_threshold_mask(img, foreground_polarity=polarity)
        dice, iou = overlap_scores(pred, ref)
        c_dice, c_iou = overlap_scores(np.ones_like(pred, dtype=bool), ref)
        rows.append(
            {
                "sample_id": rec["sample_id"],
                "category": rec["category"],
                "group_id": rec["group_id"],
                "foreground_polarity": polarity,
                "otsu_threshold": thr,
                "otsu_dice": dice,
                "otsu_iou": iou,
                "all_foreground_dice": c_dice,
                "all_foreground_iou": c_iou,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluated {len(rows)} validation images | Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
