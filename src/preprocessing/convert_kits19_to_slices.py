import os
import csv

import nibabel as nib
import numpy as np
import cv2

# -------- Paths --------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

KITS_DATA_DIR = os.path.join(ROOT, "kits19", "data")

IMG_OUT_DIR = os.path.join(ROOT, "data", "raw", "ct_slices")
MASK_OUT_DIR = os.path.join(ROOT, "data", "annotations", "ct_masks")
LABELS_CSV = os.path.join(ROOT, "data", "raw", "labels.csv")

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(MASK_OUT_DIR, exist_ok=True)

# -------- Helpers --------

def normalize_ct_slice(img: np.ndarray) -> np.ndarray:
    """
    High-quality CT normalization:
    - Clip extreme values (1st–99th percentile)
    - Scale to 0–255 uint8
    This helps the network train stably and improves accuracy.
    """
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)

    img = img - img.min()
    img = img / (img.max() + 1e-8)
    img = (img * 255.0).astype("uint8")
    return img


def process_case(case_path: str, case_name: str, labels_list: list):
    vol_path = os.path.join(case_path, "imaging.nii.gz")
    seg_path = os.path.join(case_path, "segmentation.nii.gz")

    if not os.path.exists(vol_path) or not os.path.exists(seg_path):
        print(f"[SKIP] {case_name}: missing imaging or segmentation file")
        return

    vol = nib.load(vol_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    num_slices = vol.shape[2]
    used_slices = 0

    for z in range(num_slices):
        img_slice = vol[:, :, z]
        mask_slice = seg[:, :, z]

        # Skip slices with no kidney/tumor (no useful info for training)
        if mask_slice.max() == 0:
            continue

        img_norm = normalize_ct_slice(img_slice)

        # 256x256: good trade-off between detail and speed
        img_resized = cv2.resize(img_norm, (256, 256))
        mask_bin = (mask_slice > 0).astype("uint8") * 255
        mask_resized = cv2.resize(
            mask_bin, (256, 256), interpolation=cv2.INTER_NEAREST
        )

        filename = f"{case_name}_z{z:03d}.png"
        img_path = os.path.join(IMG_OUT_DIR, filename)
        mask_path = os.path.join(MASK_OUT_DIR, filename.replace(".png", "_mask.png"))

        cv2.imwrite(img_path, img_resized)
        cv2.imwrite(mask_path, mask_resized)

        # CKD stage not assigned yet → placeholder -1
        labels_list.append((filename, -1))
        used_slices += 1

    print(f"[OK] {case_name}: saved {used_slices} useful slices")


def main():
    case_dirs = sorted(
        d for d in os.listdir(KITS_DATA_DIR)
        if d.startswith("case_") and os.path.isdir(os.path.join(KITS_DATA_DIR, d))
    )

    labels = [("filename", "ckd_stage")]  # header for labels.csv

    for case_name in case_dirs:
        case_path = os.path.join(KITS_DATA_DIR, case_name)
        process_case(case_path, case_name, labels)

    # Write labels.csv
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(labels)

    print(f"\nDone. Total slices: {len(labels) - 1}")
    print(f"Images saved to: {IMG_OUT_DIR}")
    print(f"Masks  saved to: {MASK_OUT_DIR}")
    print(f"labels.csv written to: {LABELS_CSV}")


if __name__ == "__main__":
    main()
