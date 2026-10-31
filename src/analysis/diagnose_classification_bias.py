"""
Diagnostic Script: Analyze Model Predictions
Checks if model is biased toward certain stages.
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.data.ckd_classification_dataset import CKDClassificationDataset
from src.models.efficientnet_ckd import EfficientNetCKD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROI_DIR = os.path.join(ROOT, "data", "classification", "kidney_rois")
LABELS_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd_balanced.csv")
MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_v2.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def main():
    print("=" * 70)
    print("Model Prediction Bias Analysis")
    print("=" * 70)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = CKDClassificationDataset(ROI_DIR, LABELS_CSV)
    print(f"  Total samples: {len(dataset)}")
    
    # Check actual distribution
    actual_stages = [stage for _, stage in dataset.samples]
    actual_dist = Counter(actual_stages)
    print(f"\n  Actual class distribution:")
    for stage in sorted(actual_dist.keys()):
        print(f"    Stage {stage}: {actual_dist[stage]} ({100*actual_dist[stage]/len(dataset):.1f}%)")
    
    # Load model
    print(f"\n[2/4] Loading model from: {MODEL_PATH}")
    model = EfficientNetCKD(num_classes=4, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # Create dataloader
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False)
    
    # Get predictions
    print(f"\n[3/4] Running inference on {len(dataset)} samples...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds.tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")
    
    # Convert to stages (0-3 -> 1-4)
    all_preds_stages = [p + 1 for p in all_preds]
    all_labels_stages = [l + 1 for l in all_labels]
    
    # Analyze predictions
    print(f"\n[4/4] Analysis Results:")
    print("=" * 70)
    
    pred_dist = Counter(all_preds_stages)
    print(f"\nPredicted class distribution:")
    for stage in sorted(pred_dist.keys()):
        print(f"  Stage {stage}: {pred_dist[stage]} ({100*pred_dist[stage]/len(all_preds_stages):.1f}%)")
    
    # Confusion matrix
    print(f"\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
    confusion = np.zeros((4, 4), dtype=int)
    for true_stage, pred_stage in zip(all_labels_stages, all_preds_stages):
        confusion[true_stage - 1, pred_stage - 1] += 1
    
    print("      ", end="")
    for s in range(1, 5):
        print(f"Stage {s:2d}", end="  ")
    print()
    for true_s in range(1, 5):
        print(f"Stage {true_s}:", end=" ")
        for pred_s in range(1, 5):
            count = confusion[true_s - 1, pred_s - 1]
            print(f"{count:5d}", end=" ")
        print()
    
    # Per-class accuracy
    print(f"\nPer-class Accuracy:")
    for stage in range(1, 5):
        true_positives = confusion[stage - 1, stage - 1]
        total_actual = sum(confusion[stage - 1, :])
        acc = true_positives / total_actual if total_actual > 0 else 0
        print(f"  Stage {stage}: {acc:.3f} ({true_positives}/{total_actual})")
    
    # Overall accuracy
    correct = sum(all_preds_stages[i] == all_labels_stages[i] for i in range(len(all_preds_stages)))
    overall_acc = correct / len(all_preds_stages)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    
    # Check for bias
    print(f"\n{'='*70}")
    print("Bias Analysis:")
    print("=" * 70)
    
    stage_3_4_ratio = (pred_dist[3] + pred_dist[4]) / len(all_preds_stages)
    stage_1_2_ratio = (pred_dist[1] + pred_dist[2]) / len(all_preds_stages)
    
    print(f"Predictions for Stage 3+4: {stage_3_4_ratio:.1%}")
    print(f"Predictions for Stage 1+2: {stage_1_2_ratio:.1%}")
    print(f"Expected (balanced): 50% each")
    
    if stage_3_4_ratio > 0.65:
        print(f"\n⚠️  STRONG BIAS detected: Model predicts Stage 3/4 too often!")
        print(f"   This suggests:")
        print(f"   1. Model learned wrong patterns")
        print(f"   2. Need class weights or focal loss")
        print(f"   3. May need to retrain with better loss function")
    elif stage_3_4_ratio > 0.55:
        print(f"\n⚠️  MILD BIAS detected: Model slightly favors Stage 3/4")
    else:
        print(f"\n[OK] No significant bias detected")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

