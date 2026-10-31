"""
Focal Loss Training Script - Better for Class Imbalance
Uses Focal Loss to focus on hard examples and reduce bias.
"""
import os
import sys
import time
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.data.ckd_classification_dataset import CKDClassificationDataset
from src.models.efficientnet_ckd import EfficientNetCKD
ROI_DIR = os.path.join(ROOT, "data", "classification", "kidney_rois")
CLEAN_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd_balanced_clean.csv")
BALANCED_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd_balanced.csv")
LABELS_CSV = CLEAN_CSV if os.path.isfile(CLEAN_CSV) else BALANCED_CSV

MODEL_DIR = os.path.join(ROOT, "models_ckd")
LOG_DIR = os.path.join(ROOT, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Create log file with timestamp
LOG_FILE = os.path.join(LOG_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def log_print(message):
    """Print and log message."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")
        f.flush()

# Force GPU usage
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available! Training requires GPU.")
DEVICE = "cuda"
print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
BATCH_SIZE = 32
EPOCHS = 30
INIT_LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
FOCAL_ALPHA = 0.25  # Focal loss alpha
FOCAL_GAMMA = 2.0   # Focal loss gamma


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def get_advanced_transforms(train=True):
    if train:
        # Simplified augmentation for faster training
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])
    return A.Compose([A.Resize(224, 224)])


def accuracy(pred, target):
    return (pred.argmax(1) == target).float().mean().item()


class DatasetWrapper(torch.utils.data.Dataset):
    """Wrapper for applying transforms to dataset."""
    def __init__(self, ds, tf):
        self.ds = ds
        self.tf = tf
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        img, label = self.ds[i]
        # Convert from tensor to numpy
        img = img.squeeze(0).numpy()
        
        # Convert from float32 [0,1] to uint8 [0,255] for albumentations
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        
        # Expand to HWC format for albumentations
        img = np.expand_dims(img, -1)  # (H, W, 1)
        
        # Apply transforms (albumentations expects uint8)
        transformed = self.tf(image=img)
        img = transformed["image"]
        
        # Convert back to CHW float32 tensor [0,1]
        img = img.transpose(2, 0, 1)  # (1, H, W)
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img, dtype=torch.float32)
        
        return img, label


def estimate_training_time(dataset_size, batch_size, epochs, device="cuda"):
    secs_per_batch = 0.08 if device == "cuda" else 0.5
    batches_per_epoch = (dataset_size * 0.85) / batch_size
    total_batches = batches_per_epoch * epochs
    total_seconds = total_batches * secs_per_batch
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return {
        "total_seconds": total_seconds,
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
        "batches_per_epoch": int(batches_per_epoch),
        "total_batches": int(total_batches)
    }


def main():
    log_print("=" * 70)
    log_print("Focal Loss Training (Addresses Class Bias)")
    log_print("=" * 70)
    log_print(f"\nTraining started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Log file: {LOG_FILE}")
    log_print(f"\n[INFO] Device: {DEVICE}")
    log_print(f"[INFO] Labels CSV: {LABELS_CSV}")
    log_print(f"[INFO] Focal Loss: alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}")
    
    # Load dataset
    log_print(f"\n[1/5] Loading dataset...")
    full_dataset = CKDClassificationDataset(ROI_DIR, LABELS_CSV)
    log_print(f"  Total samples: {len(full_dataset)}")
    
    # Check distribution
    from collections import Counter
    stage_counts = Counter([stage for _, stage in full_dataset.samples])
    log_print(f"\n  Class distribution:")
    for stage in sorted(stage_counts.keys()):
        log_print(f"    Stage {stage}: {stage_counts[stage]} ({100*stage_counts[stage]/len(full_dataset):.1f}%)")
    
    # Estimate time
    log_print(f"\n[2/5] Estimating training time...")
    time_est = estimate_training_time(len(full_dataset), BATCH_SIZE, EPOCHS, DEVICE)
    log_print(f"  Estimated time: {time_est['hours']}h {time_est['minutes']}m {time_est['seconds']}s")
    log_print(f"\n[INFO] Starting training automatically...")
    
    # Split dataset
    val_len = int(0.15 * len(full_dataset))
    train_len = len(full_dataset) - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len],
                                     generator=torch.Generator().manual_seed(42))
    
    # Wrap with transforms
    train_dataset = DatasetWrapper(train_ds, get_advanced_transforms(True))
    val_dataset = DatasetWrapper(val_ds, get_advanced_transforms(False))
    
    # Windows compatibility: num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    # Initialize model
    log_print(f"\n[3/5] Initializing model...")
    model = EfficientNetCKD(num_classes=4, pretrained=True).to(DEVICE)
    log_print(f"  Model initialized on {DEVICE}")
    
    # Focal loss
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # Training loop
    log_print(f"\n[4/5] Starting training...")
    log_print(f"  Total epochs: {EPOCHS}")
    log_print(f"  Batch size: {BATCH_SIZE}")
    log_print(f"  Train batches per epoch: {len(train_loader)}")
    log_print(f"  Val batches: {len(val_loader)}")
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(MODEL_DIR, "efficientnet_ckd_best_focal.pth")
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        log_print(f"\n--- Epoch {epoch}/{EPOCHS} Starting ---")
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx == 0:
                log_print(f"  Processing batch 1/{len(train_loader)}...")
            elif (batch_idx + 1) % 100 == 0:
                log_print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches...")
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy(out, y)
            train_batches += 1
        
        train_loss /= train_batches
        train_acc /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_acc += accuracy(out, y)
                val_batches += 1
        
        val_loss /= val_batches
        val_acc /= val_batches
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        log_print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            log_print(f"  >>> BEST MODEL SAVED (Val Acc: {best_val_acc:.4f}) <<<")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            log_print(f"\n[5/5] Early stopping triggered (no improvement for {PATIENCE} epochs)")
            break
    
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    log_print(f"\n[5/5] Training Complete!")
    log_print("=" * 70)
    log_print(f"Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    log_print(f"Total Time: {hours}h {minutes}m {seconds}s")
    log_print(f"Model Saved: {best_model_path}")
    log_print("=" * 70)
    log_print(f"\nView full log at: {LOG_FILE}")


if __name__ == "__main__":
    main()

