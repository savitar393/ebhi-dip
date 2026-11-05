"""Reusable Matplotlib visualization and histogram routines for EBHI-DIP."""

from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ebhi_dip.analysis import TIE_TOLERANCE, SegmentationCase, method_label
from ebhi_dip.dataset import REFERENCE_THRESHOLD
from ebhi_dip.metrics import confusion_counts
from ebhi_dip.segmentation import rgb_to_gray, rgb_to_lab

TP_COLOR_RGB = np.array([46, 160, 67], dtype=np.float32)    # #2EA043 hijau
FP_COLOR_RGB = np.array([230, 126, 34], dtype=np.float32)   # #E67E22 oranye
FN_COLOR_RGB = np.array([192, 57, 159], dtype=np.float32)   # #C0399F magenta

CATEGORY_PALETTE: dict[str, str] = {
    "Adenocarcinoma": "#C0392B",
    "High-grade IN": "#D35400",
    "Low-grade IN": "#B7950B",
    "Normal": "#1F618D",
    "Polyp": "#1E8449",
    "Serrated adenoma": "#7D3C98",
}


def configure_notebook_style() -> None:
    """Configure consistent, readable Matplotlib defaults for notebooks and exported figures."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def finish_figure(
    fig: Figure,
    export_path: str | Path | None = None,
    show: bool = True,
    close: bool = True,
) -> Path | None:
    """Optionally export a figure to disk, display it, and close it cleanly."""
    saved: Path | None = None
    if export_path is not None:
        saved = Path(export_path)
        saved.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return saved


def build_error_overlay(
    image_rgb: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    alpha: float = 0.52,
) -> tuple[np.ndarray, dict[str, int]]:
    """Create RGB error overlay with TP=green, FP=orange, FN=magenta, TN=transparent."""
    rgb = np.asarray(image_rgb)
    pred = np.asarray(predicted, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected uint8 RGB image (H, W, 3), got dtype={rgb.dtype}, shape={rgb.shape}.")
    if pred.shape != rgb.shape[:2] or ref.shape != rgb.shape[:2]:
        raise ValueError("Prediction/reference shapes must match image spatial dimensions.")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Overlay alpha must be in [0, 1], got {alpha}.")

    counts = confusion_counts(pred, ref)
    tp_mask = pred & ref
    fp_mask = pred & ~ref
    fn_mask = ~pred & ref

    blended = rgb.astype(np.float32).copy()
    for mask, color in (
        (tp_mask, TP_COLOR_RGB),
        (fp_mask, FP_COLOR_RGB),
        (fn_mask, FN_COLOR_RGB),
    ):
        if np.any(mask):
            blended[mask] = (1.0 - alpha) * blended[mask] + alpha * color

    overlay_u8 = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return overlay_u8, counts


def case_metrics_table(case: SegmentationCase) -> pd.DataFrame:
    """Build a per-case comparison DataFrame of overlap, precision/recall, coverage, and TP/FP/FN/TN."""
    rows = []
    for m_id, m_dict in ((case.method_a, case.metrics_a), (case.method_b, case.metrics_b)):
        rows.append(
            {
                "method": m_id,
                "label": method_label(m_id),
                "dice": float(m_dict["dice"]),
                "iou": float(m_dict["iou"]),
                "pixel_precision": float(m_dict["precision"]),
                "pixel_recall": float(m_dict["recall"]),
                "predicted_fg_pct": 100.0 * float(m_dict["predicted_foreground_fraction"]),
                "reference_fg_pct": 100.0 * float(m_dict["reference_foreground_fraction"]),
                "tp": int(m_dict["tp"]),
                "fp": int(m_dict["fp"]),
                "fn": int(m_dict["fn"]),
                "tn": int(m_dict["tn"]),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def format_case_metrics_for_display(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Format case metrics table with explicit 'undefined (NaN)' for zero-denominator rates."""
    formatted = pd.DataFrame(index=metrics_df.index)
    formatted["Metode"] = metrics_df["label"]
    for col, out_name in (
        ("dice", "Dice"),
        ("iou", "IoU"),
        ("pixel_precision", "Precision"),
        ("pixel_recall", "Recall"),
    ):
        formatted[out_name] = [
            "undefined (NaN)" if math.isnan(float(v)) else f"{float(v):.4f}"
            for v in metrics_df[col]
        ]
    formatted["Pred FG (%)"] = [f"{float(v):.2f}%" for v in metrics_df["predicted_fg_pct"]]
    formatted["Ref FG (%)"] = [f"{float(v):.2f}%" for v in metrics_df["reference_fg_pct"]]
    for c_name in ("tp", "fp", "fn", "tn"):
        formatted[c_name.upper()] = metrics_df[c_name].astype(int)
    return formatted


def plot_segmentation_case(case: SegmentationCase) -> Figure:
    """Plot 2x3 paired segmentation gallery: RGB, binary ref, pred A, pred B, overlay A, overlay B."""
    overlay_a, counts_a = build_error_overlay(case.image_rgb, case.prediction_a, case.binary_reference)
    overlay_b, counts_b = build_error_overlay(case.image_rgb, case.prediction_b, case.binary_reference)
    total_px = case.binary_reference.size

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.9), layout="constrained")

    axes[0, 0].imshow(case.image_rgb, interpolation="nearest")
    axes[0, 0].set_title(f"1. Original RGB\n{case.image_rgb.shape[1]}×{case.image_rgb.shape[0]} px")

    ref_pct = 100.0 * float(case.metrics_a["reference_foreground_fraction"])
    axes[0, 1].imshow(case.binary_reference, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 1].set_title(
        f"2. Binary reference (mask >= {REFERENCE_THRESHOLD})\nForeground: {ref_pct:.1f}%"
    )

    pred_a_pct = 100.0 * float(case.metrics_a["predicted_foreground_fraction"])
    axes[0, 2].imshow(case.prediction_a, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 2].set_title(
        f"3. Prediksi A: {case.method_a}\n"
        f"Dice={case.metrics_a['dice']:.4f} | IoU={case.metrics_a['iou']:.4f} | FG={pred_a_pct:.1f}%"
    )

    pred_b_pct = 100.0 * float(case.metrics_b["predicted_foreground_fraction"])
    axes[1, 0].imshow(case.prediction_b, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[1, 0].set_title(
        f"4. Prediksi B: {case.method_b}\n"
        f"Dice={case.metrics_b['dice']:.4f} | IoU={case.metrics_b['iou']:.4f} | FG={pred_b_pct:.1f}%"
    )

    axes[1, 1].imshow(overlay_a, interpolation="nearest")
    axes[1, 1].set_title(
        f"5. Error overlay A ({case.method_a})\n"
        f"TP={100 * counts_a['tp'] / total_px:.1f}% | "
        f"FP={100 * counts_a['fp'] / total_px:.1f}% | "
        f"FN={100 * counts_a['fn'] / total_px:.1f}%"
    )

    axes[1, 2].imshow(overlay_b, interpolation="nearest")
    axes[1, 2].set_title(
        f"6. Error overlay B ({case.method_b})\n"
        f"TP={100 * counts_b['tp'] / total_px:.1f}% | "
        f"FP={100 * counts_b['fp'] / total_px:.1f}% | "
        f"FN={100 * counts_b['fn'] / total_px:.1f}%"
    )

    for ax in axes.ravel():
        ax.axis("off")

    reasons_text = "; ".join(case.selection_reasons) if case.selection_reasons else "Pilihan manual"
    fig.suptitle(
        f"{case.sample_id} | Kategori: {case.category} | Group: {case.group_id}\n"
        f"A={case.method_a} vs B={case.method_b} | Selisih Dice (A−B) = {case.delta_dice:+.4f} | "
        f"Alasan: {reasons_text}",
        fontsize=11.5,
        fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor="#2EA043", edgecolor="#1E6F2D", label="TP: pred & ref (hijau)"),
        Patch(facecolor="#E67E22", edgecolor="#A85913", label="FP: pred & ~ref (oranye)"),
        Patch(facecolor="#C0399F", edgecolor="#802369", label="FN: ~pred & ref (magenta)"),
        Patch(facecolor="#EAEAEA", edgecolor="#888888", label="TN: ~pred & ~ref (transparan / RGB asli)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=2,
        frameon=True,
        fontsize=9.5,
    )
    return fig


def uint8_bin_edges(bins: int = 64) -> np.ndarray:
    """Build bin edges covering all uint8 integers 0..255 inclusively."""
    if int(bins) < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")
    n_bins = min(int(bins), 256)
    if n_bins == 256:
        return np.arange(257, dtype=np.float64) - 0.5
    return np.linspace(-0.5, 255.5, n_bins + 1, dtype=np.float64)


def compute_shared_lab_edges(
    lab_arrays: Sequence[np.ndarray] | np.ndarray,
    bins: int = 64,
) -> dict[str, np.ndarray]:
    """Compute shared finite bin edges across one or more Lab arrays for L*, a*, and b*."""
    if int(bins) < 1:
        raise ValueError(f"bins must be >= 1, got {bins}.")

    if isinstance(lab_arrays, np.ndarray):
        items = [lab_arrays]
    else:
        items = list(lab_arrays)
    if not items:
        raise ValueError("At least one Lab array is required to compute shared bin edges.")

    channel_names = ("L*", "a*", "b*")
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]

    for arr in items:
        lab = np.asarray(arr, dtype=np.float64)
        if lab.ndim < 2 or lab.shape[-1] != 3:
            raise ValueError(f"Expected Lab array with last dimension 3, got shape {lab.shape}.")
        if lab.size == 0:
            raise ValueError("Lab array must be non-empty to derive shared edges.")
        if not np.isfinite(lab).all():
            raise ValueError("Lab array contains non-finite values.")
        flat = lab.reshape(-1, 3)
        for idx in range(3):
            mins[idx] = min(mins[idx], float(flat[:, idx].min()))
            maxs[idx] = max(maxs[idx], float(flat[:, idx].max()))

    edges_by_channel: dict[str, np.ndarray] = {}
    for idx, ch_name in enumerate(channel_names):
        low, high = mins[idx], maxs[idx]
        if low == high:
            low -= 0.5
            high += 0.5
        edges = np.linspace(low, high, int(bins) + 1, dtype=np.float64)
        if edges[0] > mins[idx]:
            edges[0] = np.nextafter(mins[idx], -math.inf)
        if edges[-1] < maxs[idx]:
            edges[-1] = np.nextafter(maxs[idx], math.inf)
        edges_by_channel[ch_name] = edges
    return edges_by_channel


def compute_histogram_1d(
    values: np.ndarray,
    edges: np.ndarray,
    density: bool = False,
) -> dict[str, object]:
    """Compute exact 1D histogram counts (and optional density), handling empty regions safely."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    edge_arr = np.asarray(edges, dtype=np.float64)
    if edge_arr.ndim != 1 or len(edge_arr) < 2 or not np.all(np.diff(edge_arr) > 0):
        raise ValueError("Histogram edges must be a strictly increasing 1D array of length >= 2.")
    if arr.size > 0 and not np.isfinite(arr).all():
        raise ValueError("Histogram input contains non-finite values.")

    n_bins = len(edge_arr) - 1
    if arr.size == 0:
        return {
            "edges": edge_arr,
            "counts": np.zeros(n_bins, dtype=np.int64),
            "heights": np.zeros(n_bins, dtype=np.float64),
            "pixel_count": 0,
            "empty": True,
            "density": bool(density),
        }

    counts, _ = np.histogram(arr, bins=edge_arr, density=False)
    if int(counts.sum()) != int(arr.size):
        raise ValueError(
            f"Histogram edges [{edge_arr[0]}, {edge_arr[-1]}] dropped pixels: "
            f"counted {int(counts.sum())} of {int(arr.size)} values "
            f"(min={float(arr.min())}, max={float(arr.max())})."
        )

    if density:
        widths = np.diff(edge_arr)
        heights = counts.astype(np.float64) / (float(arr.size) * widths)
    else:
        heights = counts.astype(np.float64)

    return {
        "edges": edge_arr,
        "counts": counts.astype(np.int64),
        "heights": heights,
        "pixel_count": int(arr.size),
        "empty": False,
        "density": bool(density),
    }


def compute_region_histograms(
    channel_2d: np.ndarray,
    region_masks: dict[str, np.ndarray],
    edges: np.ndarray,
    density: bool = False,
) -> dict[str, dict[str, object]]:
    """Compute post-hoc histograms conditioned on boolean region masks using shared edges."""
    ch = np.asarray(channel_2d)
    out: dict[str, dict[str, object]] = {}
    for region_name, mask in region_masks.items():
        m = np.asarray(mask, dtype=bool)
        if m.shape != ch.shape:
            raise ValueError(
                f"Region mask '{region_name}' shape {m.shape} does not match channel {ch.shape}."
            )
        out[region_name] = compute_histogram_1d(ch[m], edges=edges, density=density)
    return out


def compute_case_histograms(
    case: SegmentationCase,
    bins: int = 64,
    lab_edges: dict[str, np.ndarray] | None = None,
    density: bool = False,
) -> dict[str, object]:
    """Compute all six panel histograms for a SegmentationCase using exact pipeline conversions."""
    u8_edges = uint8_bin_edges(bins)
    gray = rgb_to_gray(case.image_rgb)
    lab = rgb_to_lab(case.image_rgb)
    resolved_lab_edges = (
        lab_edges if lab_edges is not None else compute_shared_lab_edges([lab], bins=bins)
    )

    return {
        "R": compute_histogram_1d(case.image_rgb[:, :, 0], u8_edges, density=density),
        "G": compute_histogram_1d(case.image_rgb[:, :, 1], u8_edges, density=density),
        "B": compute_histogram_1d(case.image_rgb[:, :, 2], u8_edges, density=density),
        "gray": compute_histogram_1d(gray, u8_edges, density=density),
        "reference_raw": compute_histogram_1d(case.raw_reference_l, u8_edges, density=density),
        "L*": compute_histogram_1d(lab[:, :, 0], resolved_lab_edges["L*"], density=density),
        "a*": compute_histogram_1d(lab[:, :, 1], resolved_lab_edges["a*"], density=density),
        "b*": compute_histogram_1d(lab[:, :, 2], resolved_lab_edges["b*"], density=density),
        "otsu_threshold": case.otsu_threshold,
        "otsu_polarity": case.otsu_polarity,
        "reference_threshold": REFERENCE_THRESHOLD,
        "lab_edges": resolved_lab_edges,
        "density": bool(density),
    }


def _step_from_hist(ax: plt.Axes, hist: dict[str, object], **kwargs) -> None:
    ax.stairs(hist["heights"], hist["edges"], **kwargs)


def plot_image_histograms(
    case: SegmentationCase,
    bins: int = 64,
    lab_edges: dict[str, np.ndarray] | None = None,
    density: bool = False,
) -> Figure:
    """Plot six-panel image intensity & color-space histograms for a SegmentationCase."""
    hists = compute_case_histograms(case, bins=bins, lab_edges=lab_edges, density=density)
    y_label = "Kerapatan" if density else "Jumlah piksel"

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.8), layout="constrained")

    # 1. RGB
    ax_rgb = axes[0, 0]
    for ch, color in (("R", "#C0392B"), ("G", "#1E8449"), ("B", "#2471A3")):
        _step_from_hist(ax_rgb, hists[ch], color=color, linewidth=1.4, label=ch)
    ax_rgb.set_xlim(-0.5, 255.5)
    ax_rgb.set_title("1. RGB original (uint8, 0–255)")
    ax_rgb.set_xlabel("Intensitas channel (0–255)")
    ax_rgb.set_ylabel(y_label)
    ax_rgb.legend(frameon=False, fontsize=9)
    ax_rgb.grid(axis="y", color="#EAEAEA", linewidth=0.6)

    # 2. Grayscale + Otsu threshold
    ax_gray = axes[0, 1]
    _step_from_hist(ax_gray, hists["gray"], color="#34495E", fill=True, alpha=0.28)
    _step_from_hist(ax_gray, hists["gray"], color="#2C3E50", linewidth=1.3)
    thr = float(case.otsu_threshold)
    polarity_rule = f"gray <= {thr:.0f}" if case.otsu_polarity == "dark" else f"gray > {thr:.0f}"
    ax_gray.axvline(
        thr,
        color="#C0392B",
        linestyle="--",
        linewidth=1.6,
        label=f"Otsu = {thr:.0f}\nFG ({case.otsu_polarity}): {polarity_rule}",
    )
    ax_gray.set_xlim(-0.5, 255.5)
    ax_gray.set_title("2. Histogram grayscale (input Otsu)")
    ax_gray.set_xlabel("Intensitas grayscale (0–255)")
    ax_gray.set_ylabel(y_label)
    ax_gray.legend(frameon=True, fontsize=8.5, loc="best")
    ax_gray.grid(axis="y", color="#EAEAEA", linewidth=0.6)

    # 3. Raw grayscale reference mask + fixed 128 threshold
    ax_ref = axes[0, 2]
    _step_from_hist(ax_ref, hists["reference_raw"], color="#5D6D7E", fill=True, alpha=0.30)
    _step_from_hist(ax_ref, hists["reference_raw"], color="#2E4053", linewidth=1.3)
    ax_ref.axvline(
        REFERENCE_THRESHOLD,
        color="#8E44AD",
        linestyle="-.",
        linewidth=1.6,
        label=f"Biner ref: mask >= {REFERENCE_THRESHOLD}\n(bukan threshold Otsu citra)",
    )
    ax_ref.set_xlim(-0.5, 255.5)
    ax_ref.set_title("3. Encoding raw reference mask (0–255)")
    ax_ref.set_xlabel("Nilai piksel mask mentah (0–255)")
    ax_ref.set_ylabel(y_label)
    ax_ref.legend(frameon=True, fontsize=8.5, loc="best")
    ax_ref.grid(axis="y", color="#EAEAEA", linewidth=0.6)

    # 4–6. Physical Lab L*, a*, b*
    lab_specs = (
        (axes[1, 0], "L*", "#566573", "4. Physical Lab L* (kecerahan)", "L* (float fisik, ~0–100)"),
        (axes[1, 1], "a*", "#B03A2E", "5. Physical Lab a* (hijau–merah)", "a* signed (float fisik)"),
        (axes[1, 2], "b*", "#1F618D", "6. Physical Lab b* (biru–kuning)", "b* signed (float fisik)"),
    )
    for ax_lab, ch_key, color, title, xlabel in lab_specs:
        h_ch = hists[ch_key]
        _step_from_hist(ax_lab, h_ch, color=color, fill=True, alpha=0.25)
        _step_from_hist(ax_lab, h_ch, color=color, linewidth=1.35)
        edges = h_ch["edges"]
        ax_lab.set_xlim(float(edges[0]), float(edges[-1]))
        if ch_key in ("a*", "b*") and float(edges[0]) <= 0.0 <= float(edges[-1]):
            ax_lab.axvline(0.0, color="#888888", linestyle=":", linewidth=1.0, label="Titik nol (0)")
            ax_lab.legend(frameon=False, fontsize=8.5, loc="best")
        ax_lab.set_title(title)
        ax_lab.set_xlabel(xlabel)
        ax_lab.set_ylabel(y_label)
        ax_lab.grid(axis="y", color="#EAEAEA", linewidth=0.6)

    mode_note = (
        "Mode kerapatan: integral luas histogram = 1 (bukan persentase)."
        if density
        else f"Total per channel = {case.binary_reference.size:,} piksel."
    )
    fig.suptitle(
        f"Histogram piksel per kasus: {case.sample_id} ({case.category})\n"
        f"{mode_note} Lab K-means mengelompokkan vektor 3D (L*, a*, b*), bukan satu garis threshold 1D.",
        fontsize=11,
        fontweight="bold",
    )
    return fig


def plot_region_lab_histograms(
    case: SegmentationCase,
    bins: int = 48,
    lab_edges: dict[str, np.ndarray] | None = None,
    density: bool = False,
) -> Figure:
    """Plot post-hoc Lab distributions for foreground regions of reference, method A, and method B."""
    lab = rgb_to_lab(case.image_rgb)
    edges_map = lab_edges if lab_edges is not None else compute_shared_lab_edges([lab], bins=bins)
    regions = {
        "Ref FG": case.binary_reference,
        f"{case.method_a} FG": case.prediction_a,
        f"{case.method_b} FG": case.prediction_b,
    }
    colors = {
        "Ref FG": "#2C3E50",
        f"{case.method_a} FG": "#2471A3",
        f"{case.method_b} FG": "#D35400",
    }
    y_label = "Kerapatan" if density else "Jumlah piksel"

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), layout="constrained")
    for idx, ch_name in enumerate(("L*", "a*", "b*")):
        ax = axes[idx]
        reg_hists = compute_region_histograms(
            lab[:, :, idx],
            region_masks=regions,
            edges=edges_map[ch_name],
            density=density,
        )
        for reg_label, h_reg in reg_hists.items():
            px_n = int(h_reg["pixel_count"])
            suffix = "kosong (n=0)" if h_reg["empty"] else f"n={px_n:,}"
            _step_from_hist(
                ax,
                h_reg,
                color=colors[reg_label],
                linewidth=1.5,
                label=f"{reg_label} ({suffix})",
            )
        ax.set_title(f"Distribusi foreground {ch_name}")
        ax.set_xlabel(f"{ch_name} (float fisik)")
        ax.set_ylabel(y_label)
        ax.grid(axis="y", color="#EAEAEA", linewidth=0.6)
        if idx == 0:
            ax.legend(frameon=True, fontsize=8.5, loc="best")

    fig.suptitle(
        f"Diagnostik post-hoc region foreground Lab: {case.sample_id} (bukan input inferensi)",
        fontsize=11,
        fontweight="bold",
    )
    return fig


def plot_paired_scatter(
    paired: pd.DataFrame,
    tie_tolerance: float = TIE_TOLERANCE,
) -> Figure:
    """Plot paired per-image Dice scatter (method A vs method B) on [0, 1] with identity line."""
    if paired.empty:
        raise ValueError("Paired DataFrame must be non-empty for scatter plotting.")

    method_a = str(paired["method_a"].iloc[0])
    method_b = str(paired["method_b"].iloc[0])
    n_obs = len(paired)
    n_a = int((paired["delta_dice"] > tie_tolerance).sum())
    n_b = int((paired["delta_dice"] < -tie_tolerance).sum())
    n_tie = int((paired["delta_dice"].abs() <= tie_tolerance).sum())

    fig, ax = plt.subplots(figsize=(6.8, 6.4), layout="constrained")
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#444444", linestyle="--", linewidth=1.2, label="Garis identitas (A = B)")

    for cat in sorted(paired["category"].unique()):
        sub = paired.loc[paired["category"].eq(cat)]
        ax.scatter(
            sub["dice_a"],
            sub["dice_b"],
            s=28,
            alpha=0.78,
            color=CATEGORY_PALETTE.get(cat, "#555555"),
            edgecolors="white",
            linewidths=0.35,
            label=f"{cat} (n={len(sub)})",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Dice A: {method_label(method_a)}")
    ax.set_ylabel(f"Dice B: {method_label(method_b)}")
    ax.set_title(
        f"Paired Dice per citra (n={n_obs} citra validasi)\n"
        f"{method_a} unggul: {n_a} | {method_b} unggul: {n_b} | Tie: {n_tie}",
        fontsize=11.5,
    )
    ax.grid(color="#E6E6E6", linewidth=0.6)
    ax.legend(loc="lower left", frameon=True, fontsize=8.8)
    return fig


def plot_score_histograms(
    paired: pd.DataFrame,
    bins: int = 20,
    category_label: str | None = None,
) -> Figure:
    """Plot paired Dice distributions on [0, 1] and signed delta-Dice distribution across images."""
    if paired.empty:
        raise ValueError("Paired DataFrame must be non-empty for score histograms.")
    if int(bins) < 2:
        raise ValueError(f"bins must be >= 2, got {bins}.")

    method_a = str(paired["method_a"].iloc[0])
    method_b = str(paired["method_b"].iloc[0])
    n_obs = len(paired)
    cat_note = category_label or (
        paired["category"].iloc[0] if paired["category"].nunique() == 1 else "Semua 6 kategori"
    )

    dice_edges = np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)
    hist_a = compute_histogram_1d(paired["dice_a"].to_numpy(dtype=np.float64), dice_edges)
    hist_b = compute_histogram_1d(paired["dice_b"].to_numpy(dtype=np.float64), dice_edges)

    deltas = paired["delta_dice"].to_numpy(dtype=np.float64)
    d_min = min(float(deltas.min()), -0.02)
    d_max = max(float(deltas.max()), 0.02)
    pad = 0.02 * (d_max - d_min)
    delta_edges = np.linspace(d_min - pad, d_max + pad, int(bins) + 1, dtype=np.float64)
    hist_delta = compute_histogram_1d(deltas, delta_edges)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), layout="constrained")

    ax0 = axes[0]
    _step_from_hist(ax0, hist_a, color="#2471A3", fill=True, alpha=0.25)
    _step_from_hist(
        ax0,
        hist_a,
        color="#2471A3",
        linewidth=1.6,
        label=f"A: {method_label(method_a)} (mean={paired['dice_a'].mean():.4f})",
    )
    _step_from_hist(ax0, hist_b, color="#D35400", fill=True, alpha=0.25)
    _step_from_hist(
        ax0,
        hist_b,
        color="#D35400",
        linewidth=1.6,
        label=f"B: {method_label(method_b)} (mean={paired['dice_b'].mean():.4f})",
    )
    ax0.set_xlim(0.0, 1.0)
    ax0.set_xlabel("Skor Dice per citra [0, 1]")
    ax0.set_ylabel("Jumlah citra")
    ax0.set_title("Distribusi skor Dice kedua metode")
    ax0.legend(frameon=True, fontsize=9, loc="upper left")
    ax0.grid(axis="y", color="#EAEAEA", linewidth=0.6)

    ax1 = axes[1]
    _step_from_hist(ax1, hist_delta, color="#5D6D7E", fill=True, alpha=0.32)
    _step_from_hist(ax1, hist_delta, color="#2C3E50", linewidth=1.5)
    ax1.axvline(0.0, color="#C0392B", linestyle="--", linewidth=1.5, label="ΔDice = 0 (setara)")
    ax1.set_xlim(float(delta_edges[0]), float(delta_edges[-1]))
    ax1.set_xlabel(f"Selisih Dice (ΔDice = {method_a} − {method_b})\n← {method_b} lebih tinggi | {method_a} lebih tinggi →")
    ax1.set_ylabel("Jumlah citra")
    ax1.set_title("Distribusi selisih Dice berpasangan")
    ax1.legend(handles=[Line2D([0], [0], color="#C0392B", linestyle="--", linewidth=1.5, label="ΔDice = 0")], frameon=True, fontsize=9)
    ax1.grid(axis="y", color="#EAEAEA", linewidth=0.6)

    fig.suptitle(
        f"Histogram performa per citra (n={n_obs} citra validasi | Filter: {cat_note})",
        fontsize=11.5,
        fontweight="bold",
    )
    return fig
