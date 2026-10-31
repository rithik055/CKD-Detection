import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data.ckd_classification_dataset import CKDClassificationDataset  # noqa: E402
from src.models.efficientnet_ckd import EfficientNetCKD  # noqa: E402


ROI_DIR = os.path.join(ROOT, "data", "classification", "kidney_rois")
LABELS_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd_balanced.csv")
MODEL_PATH = os.path.join(ROOT, "models_ckd", "efficientnet_ckd_best_focal.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64

OUT_DIR = os.path.join(ROOT, "Evaluation", "plots")
os.makedirs(OUT_DIR, exist_ok=True)


plt.style.use("seaborn-v0_8-darkgrid")


def _load_dataset() -> CKDClassificationDataset:
    ds = CKDClassificationDataset(ROI_DIR, LABELS_CSV)
    if len(ds) == 0:
        raise RuntimeError(f"No samples found in dataset with ROI_DIR={ROI_DIR}")
    return ds


def _load_model(num_classes: int = 4) -> EfficientNetCKD:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at: {MODEL_PATH}")
    model = EfficientNetCKD(num_classes=num_classes, pretrained=False).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def _gather_predictions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: (N,) in {0,1,2,3}
      y_pred: (N,) in {0,1,2,3}
      y_proba: (N,4) softmax probabilities
    """
    ds = _load_dataset()
    model = _load_model(num_classes=4)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            probs = F.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)

            all_labels.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_proba = np.concatenate(all_probs, axis=0)

    return y_true, y_pred, y_proba


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[1, 2, 3, 4],
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix (Stages 1–4)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix to: {out_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: str,
) -> None:
    """
    Multi-class ROC curves (one-vs-rest + micro-average).
    """
    num_classes = y_proba.shape[1]
    y_true_ovr = np.eye(num_classes)[y_true]  # (N, C) one-hot

    fig, ax = plt.subplots(figsize=(7, 5.5))

    colors = ["#22c55e", "#38bdf8", "#a855f7", "#f97316"]
    for c in range(num_classes):
        RocCurveDisplay.from_predictions(
            y_true_ovr[:, c],
            y_proba[:, c],
            name=f"Stage {c + 1}",
            ax=ax,
            color=colors[c],
        )

    RocCurveDisplay.from_predictions(
        y_true_ovr.ravel(),
        y_proba.ravel(),
        name="Micro-average",
        ax=ax,
        linestyle="--",
        color="#e5e7eb",
    )

    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[INFO] Saved ROC curves to: {out_path}")


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: str,
) -> None:
    """
    Multi-class Precision–Recall curves (one-vs-rest + micro-average).
    """
    num_classes = y_proba.shape[1]
    y_true_ovr = np.eye(num_classes)[y_true]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["#22c55e", "#38bdf8", "#a855f7", "#f97316"]

    for c in range(num_classes):
        PrecisionRecallDisplay.from_predictions(
            y_true_ovr[:, c],
            y_proba[:, c],
            name=f"Stage {c + 1}",
            ax=ax,
            color=colors[c],
        )

    PrecisionRecallDisplay.from_predictions(
        y_true_ovr.ravel(),
        y_proba.ravel(),
        name="Micro-average",
        ax=ax,
        linestyle="--",
        color="#e5e7eb",
    )

    ax.set_title("Precision–Recall Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[INFO] Saved Precision–Recall curves to: {out_path}")


def plot_pr_f1_bars(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
) -> None:
    """
    Bar chart for per-class Precision, Recall, F1-score.
    """
    labels = [0, 1, 2, 3]
    stage_names = [f"Stage {k}" for k in [1, 2, 3, 4]]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#38bdf8")
    ax.bar(x, recall, width, label="Recall", color="#22c55e")
    ax.bar(x + width, f1, width, label="F1-score", color="#a855f7")

    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1-score", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncols=3)

    for i, s in enumerate(support):
        ax.text(
            x[i],
            1.03,
            f"n={s}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#e5e7eb",
        )

    fig.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved Precision/Recall/F1 bar chart to: {out_path}")


def main() -> None:
    print("[INFO] Gathering predictions for evaluation…")
    y_true, y_pred, y_proba = _gather_predictions()

    # Text summary: precision / recall / F1 per stage + overall accuracy
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=[f"Stage {k}" for k in [1, 2, 3, 4]],
        digits=3,
    )
    print("\n=== Classification Report (Precision / Recall / F1) ===")
    print(report)

    # Plots
    plot_confusion(
        y_true,
        y_pred,
        out_path=os.path.join(OUT_DIR, "confusion_matrix.png"),
    )
    plot_roc_curves(
        y_true,
        y_proba,
        out_path=os.path.join(OUT_DIR, "roc_curves.png"),
    )
    plot_precision_recall_curves(
        y_true,
        y_proba,
        out_path=os.path.join(OUT_DIR, "precision_recall_curves.png"),
    )
    plot_pr_f1_bars(
        y_true,
        y_pred,
        out_path=os.path.join(OUT_DIR, "precision_recall_f1_bars.png"),
    )


if __name__ == "__main__":
    main()

