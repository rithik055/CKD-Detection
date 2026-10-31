import os
import glob
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.unet import UNet
from src.models.efficientnet_ckd import EfficientNetCKD


# -------------------- PATHS & CONFIG --------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Where your original CT slices are stored
CT_SLICE_DIR = os.path.join(ROOT, "data", "raw", "ct_slices")

# Where to save overlays and ROIs for demo
OUT_DIR = os.path.join(ROOT, "data", "inference")
OVERLAY_DIR = os.path.join(OUT_DIR, "overlays")
ROI_DIR = os.path.join(OUT_DIR, "rois")
os.makedirs(OVERLAY_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)

# Trained models - Use focal model (best accuracy: 96.07%)
FOCAL_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_focal.pth")
ENHANCED_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_enhanced.pth")
V2_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_v2.pth")

SEG_MODEL_PATH = os.path.join(ROOT, "models_ckd", "unet_kidney_10k_30ep.pth")


def _resolve_cls_model_path() -> str:
    """Pick the best available classifier weights with clear error if missing."""
    for candidate in (FOCAL_MODEL_PATH, ENHANCED_MODEL_PATH, V2_MODEL_PATH):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No EfficientNet CKD weights found. Expected one of: "
        f"{FOCAL_MODEL_PATH}, {ENHANCED_MODEL_PATH}, {V2_MODEL_PATH}"
    )


CLS_MODEL_PATH = _resolve_cls_model_path()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE_SEG = 512   # size used in segmentation training
IMG_SIZE_CLS = 224   # size used in EfficientNet training


# -------------------- UTILS --------------------
def load_unet_model() -> UNet:
    model = UNet(in_channels=1, out_channels=1, base_ch=64)
    state = torch.load(SEG_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Loaded UNet model from {SEG_MODEL_PATH}")
    return model


def load_cls_model(num_classes: int = 4) -> EfficientNetCKD:
    model = EfficientNetCKD(num_classes=num_classes, pretrained=False)
    state = torch.load(CLS_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Loaded EfficientNet CKD model from {CLS_MODEL_PATH}")
    return model


def get_slices_for_case(case_id: str) -> List[str]:
    pattern = os.path.join(CT_SLICE_DIR, f"{case_id}_z*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No slices found for {case_id} in {CT_SLICE_DIR}")
    print(f"[INFO] Found {len(files)} slices for {case_id}")
    return files


def segment_slice(unet: UNet, img_gray: np.ndarray) -> np.ndarray:
    """
    img_gray: H x W, uint8 (0..255)
    returns binary mask H x W (0 or 1)
    """
    # resize to training size
    img_resized = cv2.resize(img_gray, (IMG_SIZE_SEG, IMG_SIZE_SEG),
                             interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = np.expand_dims(img_norm, axis=(0, 1))  # 1,1,H,W
    img_tensor = torch.from_numpy(img_norm).to(DEVICE)

    with torch.no_grad():
        logits = unet(img_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()

    mask_np = mask.squeeze().cpu().numpy()  # H,W
    return mask_np


def create_overlay(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    original: H,W uint8
    mask: H,W float {0,1}
    returns BGR overlay image
    """
    h, w = original.shape
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    color = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(color)
    red[:, :, 2] = 255

    overlay = color.copy()
    alpha = 0.4
    overlay[mask_resized > 0.5] = (
        (1 - alpha) * color[mask_resized > 0.5] +
        alpha * red[mask_resized > 0.5]
    )

    return overlay


def extract_roi(original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    original: H,W uint8
    mask: H,W float {0,1} (segmentation output)
    returns cropped kidney ROI as H,W uint8 (square, resized to IMG_SIZE_CLS)
    """
    h, w = original.shape
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized > 0.5).astype(np.uint8)

    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        # no kidney found, return center crop
        print("[WARN] No kidney pixels found, using center crop ROI.")
        cy, cx = h // 2, w // 2
        size = min(h, w) // 2
        y1, y2 = max(0, cy - size), min(h, cy + size)
        x1, x2 = max(0, cx - size), min(w, cx + size)
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        # expand a bit
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)

    roi = original[y1:y2 + 1, x1:x2 + 1]  # H_roi, W_roi

    # make square by padding
    H, W_ = roi.shape
    side = max(H, W_)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - H) // 2
    x_off = (side - W_) // 2
    square[y_off:y_off + H, x_off:x_off + W_] = roi

    # resize to classifier size
    roi_resized = cv2.resize(square, (IMG_SIZE_CLS, IMG_SIZE_CLS),
                             interpolation=cv2.INTER_LINEAR)
    return roi_resized


def classify_rois(cls_model: EfficientNetCKD, rois: List[np.ndarray]) -> Tuple[List[int], List[np.ndarray]]:
    """
    rois: list of H,W uint8 grayscale images
    returns:
      - per-slice predicted stages (1..4)
      - per-slice probabilities (N, num_classes)
    """
    if len(rois) == 0:
        return [], []

    # build batch
    batch = []
    for roi in rois:
        roi_f = roi.astype(np.float32) / 255.0  # 0..1
        roi_f = np.expand_dims(roi_f, axis=0)   # 1,H,W
        batch.append(roi_f)

    batch_np = np.stack(batch, axis=0)  # N,1,H,W
    batch_tensor = torch.from_numpy(batch_np).to(DEVICE)

    with torch.no_grad():
        logits = cls_model(batch_tensor)
        probs = F.softmax(logits, dim=1)
        preds_idx = probs.argmax(dim=1).cpu().numpy()
        probs_np = probs.cpu().numpy()

    # map idx {0,1,2,3} -> stage {1,2,3,4}
    stages = (preds_idx + 1).tolist()
    return stages, probs_np


# -------------------- MAIN PIPELINE --------------------
def run_inference_for_case(case_id: str):
    print(f"\n[INFO] Running full pipeline for {case_id}")
    unet = load_unet_model()
    cls_model = load_cls_model(num_classes=4)

    slice_files = get_slices_for_case(case_id)

    per_slice_stages: List[int] = []
    per_slice_files: List[str] = []

    rois_for_case: List[np.ndarray] = []

    for slice_path in slice_files:
        fname = os.path.basename(slice_path)

        # 1) load slice
        img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read {slice_path}, skipping.")
            continue

        # 2) segmentation
        mask = segment_slice(unet, img)

        # 3) overlay and save
        overlay = create_overlay(img, mask)
        overlay_path = os.path.join(OVERLAY_DIR, f"{case_id}_{fname}_overlay.png")
        cv2.imwrite(overlay_path, overlay)

        # 4) ROI extraction
        roi = extract_roi(img, mask)
        roi_path = os.path.join(ROI_DIR, f"{case_id}_{fname}_roi.png")
        cv2.imwrite(roi_path, roi)

        rois_for_case.append(roi)
        per_slice_files.append(fname)

    if not rois_for_case:
        print(f"[ERROR] No valid ROIs extracted for {case_id}.")
        return

    # 5) classification for all ROIs of this case
    stages, probs = classify_rois(cls_model, rois_for_case)

    # 6) aggregate per-slice predictions -> case-level stage (majority vote)
    from collections import Counter
    counter = Counter(stages)
    case_stage = counter.most_common(1)[0][0]

    print(f"\n[RESULT] Case {case_id} predicted CKD stage: {case_stage}")
    print(f"Per-slice stage distribution: {counter}")

    # 7) optional: save a small summary text file
    summary_path = os.path.join(OUT_DIR, f"{case_id}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Case ID: {case_id}\n")
        f.write(f"Predicted CKD Stage: {case_stage}\n")
        f.write(f"Per-slice prediction counts: {dict(counter)}\n")

    print(f"[INFO] Summary saved to: {summary_path}")
    print(f"[INFO] Overlays saved to: {OVERLAY_DIR}")
    print(f"[INFO] ROIs saved to: {ROI_DIR}")


if __name__ == "__main__":
    # Example: run for one case
    # Change this to any case you have, e.g. "case_00050"
    CASE_ID = "case_00002"
    run_inference_for_case(CASE_ID)
