import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(ROOT, "logs")
OUT_DIR = os.path.join(ROOT, "Evaluation", "plots")
os.makedirs(OUT_DIR, exist_ok=True)


# Use a modern, clean style
plt.style.use("seaborn-v0_8-darkgrid")


_EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+)\s*\|\s*"
    r"Train Loss:\s*([0-9.]+)\s*\|\s*Train Acc:\s*([0-9.]+)\s*\|\s*"
    r"Val Loss:\s*([0-9.]+)\s*\|\s*Val Acc:\s*([0-9.]+)"
)


def parse_log_file(path: str) -> pd.DataFrame:
    """
    Parse a training log produced by train_ckd_classifier_focal.py.

    Returns a DataFrame with:
      epoch, max_epoch, train_loss, train_acc, val_loss, val_acc
    """
    rows: List[Dict] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _EPOCH_LINE_RE.search(line)
            if not m:
                continue
            epoch, max_epoch, tr_loss, tr_acc, val_loss, val_acc = m.groups()
            rows.append(
                {
                    "epoch": int(epoch),
                    "max_epoch": int(max_epoch),
                    "train_loss": float(tr_loss),
                    "train_acc": float(tr_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                }
            )

    if not rows:
        raise ValueError(f"No epoch lines found in log: {path}")

    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    return df


def _pick_latest_log() -> str:
    if not os.path.isdir(LOG_DIR):
        raise FileNotFoundError(f"Log directory not found: {LOG_DIR}")

    candidates = [
        os.path.join(LOG_DIR, f)
        for f in os.listdir(LOG_DIR)
        if f.startswith("training_") and f.endswith(".log")
    ]
    if not candidates:
        raise FileNotFoundError(f"No training_*.log files found in {LOG_DIR}")

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def plot_curves(df: pd.DataFrame, out_prefix: str) -> None:
    epochs = df["epoch"].tolist()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, constrained_layout=True
    )

    # Loss curves
    ax1.plot(
        epochs,
        df["train_loss"],
        label="Train Loss",
        color="#22c55e",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs,
        df["val_loss"],
        label="Val Loss",
        color="#f97316",
        linewidth=2.0,
        marker="s",
        markersize=4,
    )
    _style_axes(ax1, "Training vs Validation Loss", "", "Loss")

    # Accuracy curves
    ax2.plot(
        epochs,
        df["train_acc"],
        label="Train Accuracy",
        color="#38bdf8",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    ax2.plot(
        epochs,
        df["val_acc"],
        label="Val Accuracy",
        color="#a855f7",
        linewidth=2.0,
        marker="s",
        markersize=4,
    )
    _style_axes(ax2, "Training vs Validation Accuracy", "Epoch", "Accuracy")

    for ax in (ax1, ax2):
        ax.set_xlim(min(epochs), max(epochs))

    out_path = os.path.join(OUT_DIR, f"{out_prefix}_loss_accuracy.png")
    fig.savefig(out_path, dpi=180, facecolor="#050914")
    plt.close(fig)
    print(f"[INFO] Saved combined loss/accuracy curves to: {out_path}")


def main(log_path: str | None = None) -> None:
    """
    Entry point.

    - If log_path is None, automatically uses the most recent training_*.log.
    - Saves:
        * CSV with metrics next to plots
        * A single PNG with clean loss + accuracy curves
    """
    if log_path is None:
        log_path = _pick_latest_log()

    print(f"[INFO] Using log: {log_path}")
    df = parse_log_file(log_path)

    base_name = os.path.splitext(os.path.basename(log_path))[0]
    out_csv = os.path.join(OUT_DIR, f"{base_name}_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved parsed metrics to: {out_csv}")

    plot_curves(df, out_prefix=base_name)


if __name__ == "__main__":
    main()

