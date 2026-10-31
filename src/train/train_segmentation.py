import os
import random

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.segmentation_dataset import KidneySegmentationDataset
from src.models.unet import UNet

# --------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG_DIR = os.path.join(ROOT, "data", "raw", "ct_slices")
MASK_DIR = os.path.join(ROOT, "data", "annotations", "ct_masks")
MODEL_DIR = os.path.join(ROOT, "models_ckd")
os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------------------------
# TRAINING CONFIG (FINAL RUN)
# --------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8            # good balance for RTX 3050 6GB
NUM_EPOCHS = 30           # final training
LR = 1e-3
VAL_SPLIT = 0.1           # 10% validation

# limit number of slices used (for time + accuracy balance)
MAX_SLICES = 10000        # final: 10k slices


# --------------------------------------------------------------------
# LOSS FUNCTIONS
# --------------------------------------------------------------------
def dice_loss(pred, target, smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss for segmentation.
    pred: raw logits from model (N, 1, H, W)
    target: binary mask (N, 1, H, W)
    """
    pred = torch.sigmoid(pred)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


# --------------------------------------------------------------------
# DATA AUGMENTATIONS
# --------------------------------------------------------------------
def get_transforms(train: bool = True):
    """
    We use Albumentations to apply medically safe data augmentations
    such as horizontal flips, small rotations, shifts, scaling, and
    brightness/contrast variations.
    These improve the model’s robustness to positional and scanner-related
    variations while preserving the kidney anatomy.
    """
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(p=0.3),
            ],
            additional_targets={"mask": "mask"},
        )
    return None


# --------------------------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------------------------
def main():

    print(f"Using device: {DEVICE}")

    # seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ---- Collect file names ----
    all_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
    all_files.sort()
    print(f"Total slices available: {len(all_files)}")

    # Limit to MAX_SLICES for time-efficiency (10k slices)
    if len(all_files) > MAX_SLICES:
        all_files = all_files[:MAX_SLICES]
        print(f"Using only first {len(all_files)} slices for training (MAX_SLICES={MAX_SLICES}).")

    # shuffle before splitting
    random.shuffle(all_files)

    # ---- Train/Val split ----
    val_count = int(len(all_files) * VAL_SPLIT)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    print(f"Train slices: {len(train_files)}, Val slices: {len(val_files)}")

    # ---- Datasets ----
    train_dataset = KidneySegmentationDataset(
        IMG_DIR,
        MASK_DIR,
        train_files,
        transform=get_transforms(train=True),
    )

    val_dataset = KidneySegmentationDataset(
        IMG_DIR,
        MASK_DIR,
        val_files,
        transform=None,
    )

    # ---- Dataloaders (pin_memory=True for faster GPU transfer) ----
    pin_mem = True if DEVICE == "cuda" else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,     # 0 is safest on Windows
        pin_memory=pin_mem # speedup host->GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_mem
    )

    # ---- Model, loss, optimizer ----
    model = UNet(in_channels=1, out_channels=1, base_ch=64).to(DEVICE)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_dice = 0.0
    best_model_path = os.path.join(MODEL_DIR, "unet_kidney_10k_30ep.pth")

    # ------------------- EPOCH LOOP -------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            # non_blocking transfer works well with pin_memory=True and GPU
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            logits = model(imgs)
            loss_bce = bce(logits, masks)
            loss_dice = dice_loss(logits, masks)
            loss = loss_bce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # ------------------- VALIDATION -------------------
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                logits = model(imgs)
                loss_bce = bce(logits, masks)
                loss_dice = dice_loss(logits, masks)
                loss = loss_bce + loss_dice

                val_loss += loss.item() * imgs.size(0)

                pred = torch.sigmoid(logits)
                pred = (pred > 0.5).float()

                intersection = (pred * masks).sum(dim=(1, 2, 3))
                union = pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                dice = (2 * intersection + 1.0) / (union + 1.0)

                val_dice += dice.sum().item()

        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> New best model saved (Dice: {best_val_dice:.4f})")

    print("\nTraining completed!")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
