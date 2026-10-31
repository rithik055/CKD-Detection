"""
Create Clean Test Dataset for Teacher Validation
Selects 15-20 best CT slices per stage with excellent quality for demonstration.
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import shutil
from pathlib import Path
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.models.unet import UNet
from src.models.efficientnet_ckd import EfficientNetCKD

# Paths
CT_DIR = os.path.join(ROOT, "data", "raw", "ct_slices")
ROI_DIR = os.path.join(ROOT, "data", "classification", "kidney_rois")
LABELS_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd_balanced.csv")
TEST_DATASET_DIR = os.path.join(ROOT, "test_dataset")
SEG_MODEL_PATH = os.path.join(ROOT, "models_ckd", "unet_kidney_10k_30ep.pth")
CLS_MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_focal.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLES_PER_STAGE = 18  # 18 per stage = 72 total (good for testing)


def load_models():
    """Load UNet and EfficientNet models."""
    print(f"[INFO] Using device: {DEVICE}")
    
    # Load UNet
    print(f"[INFO] Loading UNet from: {SEG_MODEL_PATH}")
    unet = UNet(in_channels=1, out_channels=1, base_ch=64).to(DEVICE)
    unet.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
    unet.eval()
    
    # Load EfficientNet
    print(f"[INFO] Loading EfficientNet from: {CLS_MODEL_PATH}")
    effnet = EfficientNetCKD(num_classes=4, pretrained=False).to(DEVICE)
    effnet.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    effnet.eval()
    
    return unet, effnet


def preprocess_ct(ct_img):
    """Preprocess CT slice for UNet."""
    if ct_img.shape != (512, 512):
        ct_img = cv2.resize(ct_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img_norm = ct_img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return img_tensor, ct_img


def segment_slice(unet, ct_img):
    """Run segmentation."""
    img_tensor, ct_processed = preprocess_ct(ct_img)
    with torch.no_grad():
        logits = unet(img_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8) * 255
    return mask, prob


def extract_roi(ct_img, mask_prob, out_size=224):
    """Extract ROI from CT using mask."""
    bin_mask = (mask_prob > 0.5).astype(np.uint8)
    ys, xs = np.where(bin_mask > 0)
    
    if len(xs) == 0 or len(ys) == 0:
        h, w = ct_img.shape
        side = min(h, w) // 2
        y1, y2 = h // 2 - side // 2, h // 2 + side // 2
        x1, x2 = w // 2 - side // 2, w // 2 + side // 2
    else:
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        pad = int((y2 - y1) * 0.15)
        y1 = max(0, y1 - pad)
        y2 = min(ct_img.shape[0], y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(ct_img.shape[1], x2 + pad)
    
    # Ensure valid bounding box
    if y2 <= y1 or x2 <= x1:
        h, w = ct_img.shape
        side = min(h, w) // 2
        y1, y2 = h // 2 - side // 2, h // 2 + side // 2
        x1, x2 = w // 2 - side // 2, w // 2 + side // 2
    
    roi = ct_img[y1:y2, x1:x2]
    
    # Check if ROI is empty or too small
    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        # Fallback to center crop
        h, w = ct_img.shape
        side = min(h, w) // 2
        y1, y2 = h // 2 - side // 2, h // 2 + side // 2
        x1, x2 = w // 2 - side // 2, w // 2 + side // 2
        roi = ct_img[y1:y2, x1:x2]
    
    h, w = roi.shape
    if h == 0 or w == 0:
        # Last resort: use entire image
        roi = ct_img.copy()
        h, w = roi.shape
    
    side = max(h, w, 1)  # Ensure at least 1
    square = np.zeros((side, side), dtype=np.uint8)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = roi
    roi_resized = cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)
    return roi_resized


def classify_roi(effnet, roi):
    """Classify ROI."""
    img = roi.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = effnet(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    
    pred_stage = int(np.argmax(probs)) + 1
    confidence = float(probs[pred_stage - 1])
    return pred_stage, confidence, probs


def calculate_quality_score(ct_img, mask, roi, true_stage, pred_stage, confidence):
    """Calculate quality score for selecting best examples."""
    score = 0.0
    
    # 1. Image quality (contrast, sharpness)
    # Use float32 to avoid memory issues
    ct_img_f32 = ct_img.astype(np.float32)
    std_val = float(np.std(ct_img_f32))
    edges = cv2.Canny(ct_img, 50, 150)
    edge_density = np.count_nonzero(edges) / float(ct_img.size)
    score += min(std_val / 50.0, 1.0) * 0.25
    score += min(edge_density * 100, 1.0) * 0.15
    
    # 2. Segmentation quality (mask coverage)
    mask_coverage = np.count_nonzero(mask) / mask.size
    if 0.05 < mask_coverage < 0.4:
        score += 0.2
    else:
        score += max(0, 0.2 - abs(mask_coverage - 0.2) * 2)
    
    # 3. ROI quality
    roi_std = np.std(roi)
    if roi_std > 10:
        score += 0.1
    
    # 4. Classification correctness and confidence
    if pred_stage == true_stage:
        score += 0.3  # Correct prediction is most important
        score += confidence * 0.15  # High confidence bonus
    
    # 5. Visual appeal (no extreme brightness/darkness)
    mean_intensity = np.mean(ct_img)
    if 50 < mean_intensity < 200:
        score += 0.05
    
    return score


def create_overlay(ct_img, mask):
    """Create overlay visualization."""
    ct_color = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2BGR)
    mask_color = np.zeros_like(ct_color)
    mask_color[:, :, 2] = mask  # Red channel
    overlay = cv2.addWeighted(ct_color, 0.7, mask_color, 0.3, 0)
    return overlay


def main():
    print("=" * 70)
    print("Clean Test Dataset Creation for Teacher Validation")
    print("=" * 70)
    
    # Load models
    print("\n[1/6] Loading models...")
    unet, effnet = load_models()
    
    # Load labels
    print("\n[2/6] Loading labels...")
    import csv
    labels_dict = {}
    stage_files = defaultdict(list)
    
    with open(LABELS_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            stage = int(row["stage"])
            labels_dict[fname] = stage
            stage_files[stage].append(fname)
    
    print(f"  Total slices: {sum(len(files) for files in stage_files.values())}")
    for stage in sorted(stage_files.keys()):
        print(f"  Stage {stage}: {len(stage_files[stage])} slices")
    
    # Create test dataset directory structure
    print(f"\n[3/6] Creating test dataset directory structure...")
    os.makedirs(TEST_DATASET_DIR, exist_ok=True)
    for stage in range(1, 5):
        stage_dir = os.path.join(TEST_DATASET_DIR, f"stage_{stage}")
        os.makedirs(stage_dir, exist_ok=True)
        for subdir in ["ct_slices", "masks", "overlays", "rois"]:
            os.makedirs(os.path.join(stage_dir, subdir), exist_ok=True)
    
    # Process and score slices (sample first to speed up)
    print(f"\n[4/6] Processing slices and selecting best examples...")
    print(f"  Strategy: Sample up to 2000 slices per stage for faster processing")
    stage_scores = defaultdict(list)
    
    import time
    import random
    start_time = time.time()
    total_processed = 0
    total_to_process = 0
    
    # Calculate total slices to process (sample max 2000 per stage)
    for stage in sorted(stage_files.keys()):
        total_to_process += min(len(stage_files[stage]), 2000)
    
    print(f"  Total slices to process: {total_to_process} (sampled from {sum(len(files) for files in stage_files.values())})")
    
    for stage in sorted(stage_files.keys()):
        stage_files_list = stage_files[stage]
        # Sample up to 2000 slices per stage for faster processing
        if len(stage_files_list) > 2000:
            stage_files_list = random.sample(stage_files_list, 2000)
            print(f"\n  Processing Stage {stage} (sampled {len(stage_files_list)} from {len(stage_files[stage])} slices)...")
        else:
            print(f"\n  Processing Stage {stage} ({len(stage_files_list)} slices)...")
        
        stage_start = time.time()
        for idx, fname in enumerate(stage_files_list, 1):
            ct_path = os.path.join(CT_DIR, fname)
            if not os.path.isfile(ct_path):
                continue
            
            ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
            if ct_img is None:
                continue
            
            # Resize if needed
            if ct_img.shape != (512, 512):
                ct_img = cv2.resize(ct_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            # Run segmentation
            mask, mask_prob = segment_slice(unet, ct_img)
            
            # Extract ROI
            roi = extract_roi(ct_img, mask_prob)
            
            # Classify
            pred_stage, confidence, probs = classify_roi(effnet, roi)
            
            # Calculate quality score
            score = calculate_quality_score(ct_img, mask, roi, stage, pred_stage, confidence)
            
            stage_scores[stage].append({
                "filename": fname,
                "score": score,
                "confidence": confidence,
                "pred_stage": pred_stage,
                "correct": pred_stage == stage,
                "ct_img": ct_img,
                "mask": mask,
                "roi": roi,
                "mask_prob": mask_prob,
                "probs": probs
            })
            
            total_processed += 1
            
            # Progress updates every 50 slices with time estimates
            if total_processed % 50 == 0:
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                remaining = (total_to_process - total_processed) / rate if rate > 0 else 0
                pct = (total_processed / total_to_process * 100) if total_to_process > 0 else 0
                print(f"    Progress: {total_processed}/{total_to_process} ({pct:.1f}%) | "
                      f"Elapsed: {elapsed/60:.1f}m | "
                      f"ETA: {remaining/60:.1f}m | "
                      f"Rate: {rate:.1f} slices/sec")
            
            # Stage completion update
            if idx % 200 == 0:
                stage_elapsed = time.time() - stage_start
                stage_rate = idx / stage_elapsed if stage_elapsed > 0 else 0
                stage_remaining = (len(stage_files_list) - idx) / stage_rate if stage_rate > 0 else 0
                print(f"      Stage {stage}: {idx}/{len(stage_files_list)} | "
                      f"ETA: {stage_remaining/60:.1f}m")
    
    # Select best examples per stage
    print(f"\n[5/6] Selecting top {SAMPLES_PER_STAGE} examples per stage...")
    selected_count = 0
    
    for stage in sorted(stage_scores.keys()):
        # Sort by correctness first, then score
        sorted_candidates = sorted(
            stage_scores[stage],
            key=lambda x: (x["correct"], x["score"]),
            reverse=True
        )
        
        selected = sorted_candidates[:SAMPLES_PER_STAGE]
        print(f"\n  Stage {stage}: Selected {len(selected)} examples")
        print(f"    Correct predictions: {sum(1 for s in selected if s['correct'])}/{len(selected)}")
        print(f"    Avg confidence: {np.mean([s['confidence'] for s in selected]):.3f}")
        print(f"    Avg score: {np.mean([s['score'] for s in selected]):.3f}")
        
        # Save selected examples
        stage_dir = os.path.join(TEST_DATASET_DIR, f"stage_{stage}")
        for idx, sample in enumerate(selected, 1):
            base_name = os.path.splitext(sample["filename"])[0]
            
            # Save CT slice
            ct_path = os.path.join(stage_dir, "ct_slices", f"{idx:02d}_{base_name}_ct.png")
            cv2.imwrite(ct_path, sample["ct_img"])
            
            # Save mask
            mask_path = os.path.join(stage_dir, "masks", f"{idx:02d}_{base_name}_mask.png")
            cv2.imwrite(mask_path, sample["mask"])
            
            # Save overlay
            overlay = create_overlay(sample["ct_img"], sample["mask"])
            overlay_path = os.path.join(stage_dir, "overlays", f"{idx:02d}_{base_name}_overlay.png")
            cv2.imwrite(overlay_path, overlay)
            
            # Save ROI
            roi_path = os.path.join(stage_dir, "rois", f"{idx:02d}_{base_name}_roi.png")
            cv2.imwrite(roi_path, sample["roi"])
            
            # Save metadata
            meta_path = os.path.join(stage_dir, f"{idx:02d}_{base_name}_info.txt")
            with open(meta_path, "w") as f:
                f.write(f"Original Filename: {sample['filename']}\n")
                f.write(f"True Stage: {stage}\n")
                f.write(f"Predicted Stage: {sample['pred_stage']}\n")
                f.write(f"Confidence: {sample['confidence']:.4f}\n")
                f.write(f"Quality Score: {sample['score']:.4f}\n")
                f.write(f"Correct: {'Yes' if sample['correct'] else 'No'}\n")
                f.write(f"\nAll Stage Probabilities:\n")
                for s in range(1, 5):
                    f.write(f"  Stage {s}: {sample['probs'][s-1]:.4f}\n")
            
            selected_count += 1
    
    # Create summary and README
    summary_path = os.path.join(TEST_DATASET_DIR, "README.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Clean Test Dataset for Teacher Validation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Examples: {selected_count}\n")
        f.write(f"Samples per Stage: {SAMPLES_PER_STAGE}\n\n")
        f.write("Directory Structure:\n")
        f.write("  stage_1/, stage_2/, stage_3/, stage_4/\n")
        f.write("    - ct_slices/    : Original CT images (512x512)\n")
        f.write("    - masks/        : Segmentation masks\n")
        f.write("    - overlays/     : CT + mask overlay visualization\n")
        f.write("    - rois/         : Extracted kidney ROIs (224x224)\n")
        f.write("    - *_info.txt    : Metadata for each example\n")
        f.write("\n")
        f.write("Usage:\n")
        f.write("  1. Upload CT slices from ct_slices/ folder to the web app\n")
        f.write("  2. Verify segmentation quality in masks/ folder\n")
        f.write("  3. Check ROI extraction in rois/ folder\n")
        f.write("  4. Compare predicted stages with true stages in info files\n")
        f.write("\n")
        f.write("Quality Criteria:\n")
        f.write("  - High image quality (good contrast, sharpness)\n")
        f.write("  - Clear segmentation masks\n")
        f.write("  - Well-extracted ROIs\n")
        f.write("  - Correct classification with high confidence\n")
    
    print("\n[6/6] Test Dataset Created Successfully!")
    print("=" * 70)
    print(f"\nLocation: {TEST_DATASET_DIR}")
    print(f"Total Examples: {selected_count} ({SAMPLES_PER_STAGE} per stage)")
    print(f"\nStructure:")
    print(f"  test_dataset/")
    for stage in range(1, 5):
        print(f"    stage_{stage}/")
        print(f"      ct_slices/  ({SAMPLES_PER_STAGE} examples)")
        print(f"      masks/      ({SAMPLES_PER_STAGE} examples)")
        print(f"      overlays/   ({SAMPLES_PER_STAGE} examples)")
        print(f"      rois/       ({SAMPLES_PER_STAGE} examples)")
        print(f"      *_info.txt (metadata)")
    print(f"\n✓ Ready for teacher validation!")


if __name__ == "__main__":
    main()

