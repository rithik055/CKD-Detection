import os
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.efficientnet_ckd import EfficientNetCKD


ROOT = os.path.dirname(os.path.abspath(__file__))

# Default root that holds per-stage folders with CT slices.
# Example structure:
#   CT_SLICES/
#     stage_1/
#     stage_2/
#     stage_3/
#     stage_4/
DEFAULT_CT_STAGE_ROOT = os.path.join(ROOT, "CT_SLICES")


FOCAL_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_focal.pth")
ENHANCED_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_enhanced.pth")
V2_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_v2.pth")


def _resolve_cls_model_path() -> str:
    """Choose the best available classifier weights."""
    for candidate in (FOCAL_MODEL_PATH, ENHANCED_MODEL_PATH, V2_MODEL_PATH):
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "No EfficientNet CKD weights found. Expected one of: "
        f"{FOCAL_MODEL_PATH}, {ENHANCED_MODEL_PATH}, {V2_MODEL_PATH}"
    )


CLS_MODEL_PATH = _resolve_cls_model_path()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_classifier(num_classes: int = 4) -> EfficientNetCKD:
    model = EfficientNetCKD(num_classes=num_classes, pretrained=False).to(DEVICE)
    state = torch.load(CLS_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded classifier from {CLS_MODEL_PATH} on {DEVICE}")
    return model


def preprocess_ct_for_classifier(img_gray: np.ndarray) -> torch.Tensor:
    """
    Simple preprocessing: resize to 224x224, normalize to [0,1], add batch+channel dims.
    This matches the classifier's expected input size.
    """
    img_resized = cv2.resize(img_gray, (224, 224))
    img_f = img_resized.astype(np.float32) / 255.0
    img_f = np.expand_dims(img_f, axis=(0, 1))  # (1,1,H,W)
    tensor = torch.from_numpy(img_f).to(DEVICE)
    return tensor


def classify_stage(
    model: EfficientNetCKD, img_gray: np.ndarray
) -> Tuple[int, np.ndarray]:
    """
    Return (predicted_stage 1..4, probabilities[4])
    """
    x = preprocess_ct_for_classifier(img_gray)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx + 1, probs


def discover_stage_folders(root: str) -> Dict[int, str]:
    """
    Find subfolders whose names encode the stage, e.g. 'stage_1', 'Stage2', '1', etc.
    Returns mapping {stage_int: folder_path}.
    """
    mapping: Dict[int, str] = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CT stage root not found: {root}")

    for name in os.listdir(root):
        full_path = os.path.join(root, name)
        if not os.path.isdir(full_path):
            continue
        # Try to extract a digit 1..4 from folder name.
        m = re.search(r"[1-4]", name)
        if not m:
            continue
        stage = int(m.group(0))
        mapping[stage] = full_path

    if not mapping:
        raise RuntimeError(f"No stage folders (1–4) found under {root}")

    print("[INFO] Detected stage folders:")
    for stage, path in sorted(mapping.items()):
        print(f"  Stage {stage}: {path}")
    return mapping


def clean_stage_folders(root: str = DEFAULT_CT_STAGE_ROOT) -> None:
    """
    For each PNG in each stage folder:
      - classify with the EfficientNet CKD model
      - if predicted_stage != folder_stage, move the file to a 'misplaced' area

    NOTE:
      We MOVE mismatched slices into CT_SLICES_misplaced/ so they are removed
      from the stage folders but not permanently deleted, in case you want
      to review them manually.
    """
    stage_folders = discover_stage_folders(root)
    classifier = load_classifier(num_classes=4)

    misplaced_root = os.path.join(os.path.dirname(root), "CT_SLICES_misplaced")
    os.makedirs(misplaced_root, exist_ok=True)

    total_checked = 0
    total_misplaced = 0

    for stage, folder in sorted(stage_folders.items()):
        print(f"\n[INFO] Checking folder for Stage {stage}: {folder}")
        files: List[str] = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            print("  (no image files found)")
            continue

        for fname in files:
            src_path = os.path.join(folder, fname)
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARN] Could not read image, skipping: {src_path}")
                continue

            pred_stage, probs = classify_stage(classifier, img)
            total_checked += 1

            if pred_stage != stage:
                total_misplaced += 1
                dst_folder = os.path.join(misplaced_root, f"from_stage_{stage}")
                os.makedirs(dst_folder, exist_ok=True)
                dst_path = os.path.join(dst_folder, fname)
                os.replace(src_path, dst_path)
                print(
                    f"  [MOVE] {fname}: expected Stage {stage}, "
                    f"predicted Stage {pred_stage} -> {dst_path}"
                )

    print("\n[SUMMARY]")
    print(f"  Total images checked : {total_checked}")
    print(f"  Misplaced (moved out): {total_misplaced}")
    print(f"  Misplaced folder root: {misplaced_root}")


if __name__ == "__main__":
    # You can optionally override root via env var CT_STAGE_ROOT.
    root_dir = os.environ.get("CT_STAGE_ROOT", DEFAULT_CT_STAGE_ROOT)
    clean_stage_folders(root=root_dir)

