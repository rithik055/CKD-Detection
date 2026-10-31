import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- PATHS ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

IMG_DIR = os.path.join(ROOT, "data", "raw", "ct_slices")
MASK_DIR = os.path.join(ROOT, "data", "annotations", "ct_masks")
LABELS_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd.csv")


# ---------- UTILS ----------

def load_image(path):
    """Load grayscale image and return numpy array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def overlay_mask_on_image(img, mask, alpha=0.5):
    """
    Create an RGB overlay: CT in gray, mask in red.
    """
    # img, mask are 2D arrays
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    mask_bool = mask > 0
    overlay = img_rgb.copy()
    # red channel
    overlay[mask_bool, 0] = 255
    overlay[mask_bool, 1] = 0
    overlay[mask_bool, 2] = 0

    blended = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    return blended


# ---------- 1. VISUALIZE RANDOM SAMPLES ----------

def visualize_random_samples(n_samples=4):
    """
    Show random CT slices, masks, and overlays.
    """
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    random.shuffle(all_files)
    sample_files = all_files[:n_samples]

    cols = 3  # image, mask, overlay
    rows = n_samples

    plt.figure(figsize=(4 * cols, 3 * rows))

    for i, fname in enumerate(sample_files):
        img_path = os.path.join(IMG_DIR, fname)
        mask_path = os.path.join(MASK_DIR, fname.replace(".png", "_mask.png"))

        img = load_image(img_path)
        mask = load_image(mask_path)

        overlay = overlay_mask_on_image(img, mask)

        # CT image
        plt.subplot(rows, cols, i * cols + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"{fname}\nCT")
        plt.axis("off")

        # mask
        plt.subplot(rows, cols, i * cols + 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        # overlay
        plt.subplot(rows, cols, i * cols + 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ---------- 2. INTENSITY HISTOGRAM ----------

def plot_intensity_histogram(n_samples=500):
    """
    Plot histogram of pixel intensities for a subset of slices.
    """
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    random.shuffle(all_files)
    sample_files = all_files[:n_samples]

    pixels = []

    for fname in sample_files:
        img_path = os.path.join(IMG_DIR, fname)
        img = load_image(img_path)
        pixels.append(img.flatten())

    pixels = np.concatenate(pixels)

    plt.figure(figsize=(6, 4))
    plt.hist(pixels, bins=50, color="blue", alpha=0.7)
    plt.title("CT Intensity Distribution (sampled slices)")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------- 3. SLICE COUNT PER CASE ----------

def plot_slices_per_case():
    """
    Count how many slices each case contributes and show distribution.
    """
    df = pd.read_csv(LABELS_CSV)

    # case_id is the part before "_z"
    df["case_id"] = df["filename"].str.split("_z").str[0]

    counts = df["case_id"].value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    counts.plot(kind="bar")
    plt.title("Number of slices per case")
    plt.xlabel("Case ID")
    plt.ylabel("Slice count")
    plt.tight_layout()
    plt.show()

    print("Basic stats on slices per case:")
    print(counts.describe())


# ---------- 4. MASK COVERAGE ANALYSIS ----------

def analyze_mask_coverage(n_samples=1000):
    """
    Check how much area the mask covers in a subset of slices.
    Helps detect if there are weird empty/huge masks.
    """
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    random.shuffle(all_files)
    sample_files = all_files[:n_samples]

    coverage_ratios = []

    for fname in sample_files:
        mask_path = os.path.join(MASK_DIR, fname.replace(".png", "_mask.png"))
        mask = load_image(mask_path)
        total = mask.size
        positive = np.count_nonzero(mask)
        coverage = positive / total
        coverage_ratios.append(coverage)

    coverage_ratios = np.array(coverage_ratios)

    plt.figure(figsize=(6, 4))
    plt.hist(coverage_ratios, bins=40, color="green", alpha=0.7)
    plt.title("Mask Coverage Ratio Distribution")
    plt.xlabel("Coverage (0–1)")
    plt.ylabel("Slice count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Mask coverage stats:")
    print(pd.Series(coverage_ratios).describe())


# ---------- MAIN ----------

def main():
    print("IMG_DIR :", IMG_DIR)
    print("MASK_DIR:", MASK_DIR)
    print("LABELS_CSV:", LABELS_CSV)

    if not os.path.exists(IMG_DIR) or not os.path.exists(MASK_DIR):
        raise RuntimeError("Image or mask directory not found. Check paths.")

    print("\n1) Visualizing random CT + mask + overlay...")
    visualize_random_samples(n_samples=4)

    print("\n2) Plotting intensity histogram...")
    plot_intensity_histogram(n_samples=500)

    print("\n3) Plotting slices per case...")
    plot_slices_per_case()

    print("\n4) Analyzing mask coverage...")
    analyze_mask_coverage(n_samples=1000)


if __name__ == "__main__":
    main()
