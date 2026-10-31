import os
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.unet import UNet

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

IMG_DIR = os.path.join(ROOT, "data", "raw", "ct_slices")
MODEL_PATH = os.path.join(ROOT, "models_ckd", "unet_kidney_10k_30ep.pth")

OUT_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ---------------------------------------------------------
# IMAGE UTILITIES
# ---------------------------------------------------------
def load_ct_slice(path: str) -> np.ndarray:
    """Load CT slice as grayscale uint8 (H, W)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def preprocess_for_model(img: np.ndarray) -> torch.Tensor:
    """
    Resize, normalize, and convert to tensor (1, 1, H, W).
    Must match training preprocessing.
    """
    # resize to 256x256 (same as during training)
    img_resized = cv2.resize(img, (256, 256))

    # normalize to [0, 1]
    img_norm = img_resized.astype(np.float32) / 255.0

    # add channel and batch dims: (1, 1, H, W)
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    return img_tensor.to(DEVICE)


def predict_mask(model: torch.nn.Module, img: np.ndarray) -> np.ndarray:
    """
    Run model on single slice and return binary mask (H, W) uint8 (0 or 255).
    """
    model.eval()
    with torch.no_grad():
        x = preprocess_for_model(img)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)
        mask = (prob > 0.5).astype(np.uint8) * 255
    return mask


def make_overlay(ct_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a red overlay of the mask on the CT image.
    ct_img: grayscale (H, W)
    mask: 0/255 (H, W)
    returns: BGR image (H, W, 3)
    """
    # ensure same size
    if ct_img.shape != mask.shape:
        mask = cv2.resize(mask, (ct_img.shape[1], ct_img.shape[0]))

    ct_color = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2BGR)

    overlay = ct_color.copy()
    overlay[mask == 255] = [0, 0, 255]  # kidney region in red (BGR)

    # blend overlay with original for transparency
    alpha = 0.5
    blended = cv2.addWeighted(overlay, alpha, ct_color, 1 - alpha, 0)
    return blended


# ---------------------------------------------------------
# MAIN DEMO
# ---------------------------------------------------------
def main(num_samples: int = 4):
    # 1) Load model
    model = UNet(in_channels=1, out_channels=1, base_ch=64).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded model from: {MODEL_PATH}")

    # 2) Pick random CT slices
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    all_files.sort()
    if len(all_files) == 0:
        raise RuntimeError(f"No PNG files found in {IMG_DIR}")

    num_samples = min(num_samples, len(all_files))
    sample_files = random.sample(all_files, num_samples)
    print("Sampled slices:", sample_files)

    # 3) Create visualization grid: rows = samples, cols = 3
    rows = num_samples
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # make it 2D always

    for row, fname in enumerate(sample_files):
        img_path = os.path.join(IMG_DIR, fname)
        ct_img = load_ct_slice(img_path)

        # predict mask
        pred_mask = predict_mask(model, ct_img)

        # resize predicted mask back to original size for display
        pred_mask_resized = cv2.resize(pred_mask, (ct_img.shape[1], ct_img.shape[0]))

        # overlay
        overlay = make_overlay(ct_img, pred_mask_resized)

        # column 1: original CT
        axes[row, 0].imshow(ct_img, cmap="gray")
        axes[row, 0].set_title(f"{fname}\nCT")
        axes[row, 0].axis("off")

        # column 2: predicted mask
        axes[row, 1].imshow(pred_mask_resized, cmap="gray")
        axes[row, 1].set_title("Predicted Mask")
        axes[row, 1].axis("off")

        # column 3: overlay
        axes[row, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title("CT + Overlay")
        axes[row, 2].axis("off")

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "segmentation_results.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\nSaved visualization to: {out_path}")
    print("You can paste this image directly into PPT/report.")


if __name__ == "__main__":
    main(num_samples=4)
