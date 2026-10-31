import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROI_DIR = os.path.join(ROOT, "data", "classification", "kidney_rois")


def show_random_rois(n_samples: int = 16):
    files = [
        f for f in os.listdir(ROI_DIR)
        if f.lower().endswith(".png")
    ]

    if len(files) == 0:
        print("[ERROR] No ROI images found in:", ROI_DIR)
        return

    n_samples = min(n_samples, len(files))
    random.shuffle(files)
    samples = files[:n_samples]

    cols = 4
    rows = int(np.ceil(n_samples / cols))

    plt.figure(figsize=(12, 3 * rows))
    for i, fname in enumerate(samples, start=1):
        img_path = os.path.join(ROI_DIR, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(fname, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def check_roi_contrast(std_threshold: float = 8.0):
    files = [
        f for f in os.listdir(ROI_DIR)
        if f.lower().endswith(".png")
    ]

    total = len(files)
    if total == 0:
        print("[ERROR] No ROI images found in:", ROI_DIR)
        return

    low_contrast = 0

    for fname in files:
        img_path = os.path.join(ROI_DIR, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        std = float(np.std(img))
        if std < std_threshold:
            low_contrast += 1

    print("Total ROIs:", total)
    print(f"Low-contrast (< {std_threshold}) ROIs:", low_contrast)
    print("Percentage low-contrast: {:.2f}%".format(100.0 * low_contrast / total))


def main():
    print("[INFO] ROI folder:", ROI_DIR)
    check_roi_contrast(std_threshold=8.0)
    print("\n[INFO] Showing some random ROIs for visual inspection...")
    show_random_rois(n_samples=16)


if __name__ == "__main__":
    main()
