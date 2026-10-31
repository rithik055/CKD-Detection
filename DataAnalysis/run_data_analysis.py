"""
End-to-end data profiling for CKD CT pipeline.

This script produces a professional-style data analysis pack:
  - Random CT + mask + overlay grid
  - CT intensity histogram
  - Slices-per-case distribution
  - Mask coverage histogram
  - Random ROI grid
  - (Optional) Label / stage distribution if available in labels CSV

All figures are written to DataAnalysis/plots so you can drop them
directly into a report or slides.
"""

import os
import random
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Input data locations (reusing existing project layout)
IMG_DIR = os.path.join(ROOT, "data", "raw", "ct_slices")
MASK_DIR = os.path.join(ROOT, "data", "annotations", "ct_masks")
LABELS_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd.csv")
ROI_DIR = os.path.join(ROOT, "data", "classification", "kidney_rois")

# Output locations
OUT_DIR = os.path.join(ROOT, "DataAnalysis", "plots")
REPORT_PATH = os.path.join(ROOT, "DataAnalysis", "summary.txt")
os.makedirs(OUT_DIR, exist_ok=True)


plt.style.use("seaborn-v0_8-darkgrid")


def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _ensure_exists(paths: List[str]) -> None:
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError(f"Required path not found: {p}")


def _write_report(lines: List[str]) -> None:
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_random_ct_mask_overlays(n_samples: int = 4) -> str:
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    if not all_files:
        raise RuntimeError(f"No CT slice PNGs found in {IMG_DIR}")
    random.shuffle(all_files)
    sample_files = all_files[:n_samples]

    cols = 3
    rows = n_samples
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    for i, fname in enumerate(sample_files):
        img_path = os.path.join(IMG_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname.replace(".png", "_mask.png"))

        img = _load_image(img_path)
        mask = _load_image(mask_path)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        mask_bool = mask > 0
        overlay = img_rgb.copy()
        overlay[mask_bool, 0] = 255
        overlay[mask_bool, 1] = 0
        overlay[mask_bool, 2] = 0
        blended = cv2.addWeighted(overlay, 0.5, img_rgb, 0.5, 0)

        r = i
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].set_title(f"{fname}\nCT", fontsize=9)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(mask, cmap="gray")
        axes[r, 1].set_title("Mask", fontsize=9)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(blended)
        axes[r, 2].set_title("Overlay", fontsize=9)
        axes[r, 2].axis("off")

    fig.suptitle("Random CT slices with masks and overlays", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUT_DIR, "ct_mask_overlays.png")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def save_intensity_histogram(n_samples: int = 500) -> Tuple[str, Dict[str, float]]:
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    if not all_files:
        raise RuntimeError(f"No CT slice PNGs found in {IMG_DIR}")
    random.shuffle(all_files)
    sample_files = all_files[:n_samples]

    pixels: List[np.ndarray] = []
    for fname in sample_files:
        img_path = os.path.join(IMG_DIR, fname)
        img = _load_image(img_path)
        pixels.append(img.flatten())
    pixels_arr = np.concatenate(pixels)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pixels_arr, bins=60, color="#38bdf8", alpha=0.8)
    ax.set_title("CT Intensity Distribution (sampled slices)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "ct_intensity_histogram.png")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    stats = {
        "min": float(pixels_arr.min()),
        "max": float(pixels_arr.max()),
        "mean": float(pixels_arr.mean()),
        "std": float(pixels_arr.std()),
    }
    return out_path, stats


def save_slices_per_case() -> Tuple[str, Dict[str, float]]:
    df = pd.read_csv(LABELS_CSV)
    df["case_id"] = df["filename"].str.split("_z").str[0]
    counts = df["case_id"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    counts.plot(kind="bar", ax=ax, color="#a855f7")
    ax.set_title("Number of slices per case", fontsize=13, fontweight="bold")
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Slice count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "slices_per_case.png")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    desc = counts.describe()
    stats = {
        "cases": int(desc["count"]),
        "min_slices": float(desc["min"]),
        "max_slices": float(desc["max"]),
        "median_slices": float(desc["50%"]),
        "mean_slices": float(desc["mean"]),
    }
    return out_path, stats


def save_mask_coverage_histogram(n_samples: int = 1000) -> Tuple[str, Dict[str, float]]:
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    if not all_files:
        raise RuntimeError(f"No CT slice PNGs found in {IMG_DIR}")
    random.shuffle(all_files)
    sample_files = all_files[:n_samples]

    coverage_ratios: List[float] = []
    for fname in sample_files:
        mask_path = os.path.join(MASK_DIR, fname.replace(".png", "_mask.png"))
        mask = _load_image(mask_path)
        total = mask.size
        positive = np.count_nonzero(mask)
        coverage_ratios.append(float(positive) / float(total))

    coverage_arr = np.array(coverage_ratios, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(coverage_arr, bins=40, color="#22c55e", alpha=0.85)
    ax.set_title("Mask Coverage Ratio Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Coverage (0–1)")
    ax.set_ylabel("Slice count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "mask_coverage_histogram.png")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    desc = pd.Series(coverage_arr).describe()
    stats = {
        "samples": int(desc["count"]),
        "min": float(desc["min"]),
        "max": float(desc["max"]),
        "median": float(desc["50%"]),
        "mean": float(desc["mean"]),
    }
    return out_path, stats


def save_random_rois_grid(n_samples: int = 16) -> str:
    files = [f for f in os.listdir(ROI_DIR) if f.lower().endswith(".png")]
    if not files:
        raise RuntimeError(f"No ROI PNGs found in {ROI_DIR}")

    n_samples = min(n_samples, len(files))
    random.shuffle(files)
    samples = files[:n_samples]

    cols = 4
    rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.atleast_2d(axes)

    for i, fname in enumerate(samples):
        r = i // cols
        c = i % cols
        img_path = os.path.join(ROI_DIR, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        axes[r, c].imshow(img, cmap="gray")
        axes[r, c].set_title(fname, fontsize=8)
        axes[r, c].axis("off")

    for j in range(len(samples), rows * cols):
        r = j // cols
        c = j % cols
        axes[r, c].axis("off")

    fig.suptitle("Random Kidney ROI Samples", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(OUT_DIR, "random_rois_grid.png")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def save_stage_distribution_if_available() -> Tuple[str | None, Dict[str, float] | None]:
    """
    If labels_ckd.csv has a stage/label column, plot its distribution.
    Returns (path, stats) or (None, None) if not available.
    """
    if not os.path.isfile(LABELS_CSV):
        return None, None

    df = pd.read_csv(LABELS_CSV)
    stage_col = None
    for candidate in ("stage", "Stage", "label", "Label"):
        if candidate in df.columns:
            stage_col = candidate
            break
    if stage_col is None:
        return None, None

    counts = df[stage_col].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="#f97316")
    ax.set_title("Label / Stage Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Slice count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "stage_distribution.png")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    desc = counts.describe()
    stats = {
        "classes": int(desc["count"]),
        "min_per_class": float(desc["min"]),
        "max_per_class": float(desc["max"]),
        "median_per_class": float(desc["50%"]),
        "mean_per_class": float(desc["mean"]),
    }
    return out_path, stats


def main() -> None:
    print("[INFO] Data analysis started.")
    _ensure_exists([IMG_DIR, MASK_DIR, ROI_DIR])

    outputs: Dict[str, str] = {}
    summary_lines: List[str] = []

    outputs["ct_mask_overlays"] = save_random_ct_mask_overlays(n_samples=4)

    path_hist, stats_hist = save_intensity_histogram(n_samples=500)
    outputs["ct_intensity_histogram"] = path_hist
    summary_lines.append(
        f"CT intensity ~ mean={stats_hist['mean']:.1f}, std={stats_hist['std']:.1f}, "
        f"range=[{stats_hist['min']:.0f}, {stats_hist['max']:.0f}]"
    )

    path_slices, stats_slices = save_slices_per_case()
    outputs["slices_per_case"] = path_slices
    summary_lines.append(
        "Slices per case ~ "
        f"cases={stats_slices['cases']}, "
        f"mean={stats_slices['mean_slices']:.1f}, "
        f"median={stats_slices['median_slices']:.1f}, "
        f"range=[{stats_slices['min_slices']:.0f}, {stats_slices['max_slices']:.0f}]"
    )

    path_cov, stats_cov = save_mask_coverage_histogram(n_samples=1000)
    outputs["mask_coverage_histogram"] = path_cov
    summary_lines.append(
        "Mask coverage ~ "
        f"mean={stats_cov['mean']:.3f}, "
        f"median={stats_cov['median']:.3f}, "
        f"range=[{stats_cov['min']:.3f}, {stats_cov['max']:.3f}]"
    )

    outputs["random_rois_grid"] = save_random_rois_grid(n_samples=16)

    stage_path, stage_stats = save_stage_distribution_if_available()
    if stage_path and stage_stats:
        outputs["stage_distribution"] = stage_path
        summary_lines.append(
            "Stage distribution ~ "
            f"classes={stage_stats['classes']}, "
            f"mean per class={stage_stats['mean_per_class']:.1f}, "
            f"range per class=[{stage_stats['min_per_class']:.0f}, "
            f"{stage_stats['max_per_class']:.0f}]"
        )

    # Write a compact text summary for quick inspection / reports
    header = [
        "CKD CT DATA ANALYSIS SUMMARY",
        "============================",
        "",
    ]
    _write_report(header + summary_lines)

    print("\n[INFO] Data analysis complete. Images saved to:")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")
    print(f"\n[INFO] Summary written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

