import pandas as pd
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

LABELS_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd.csv")
BALANCED_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd_balanced.csv")

TARGET_PER_CLASS = 6000     # 6000 for each stage

def main():
    df = pd.read_csv(LABELS_CSV)
    print(df.head())

    balanced = []

    for stage in sorted(df["stage"].unique()):
        df_stage = df[df["stage"] == stage]

        if len(df_stage) >= TARGET_PER_CLASS:
            df_stage_sampled = df_stage.sample(TARGET_PER_CLASS, random_state=42)
        else:
            # oversample if fewer samples exist
            df_stage_sampled = df_stage.sample(TARGET_PER_CLASS, replace=True, random_state=42)

        balanced.append(df_stage_sampled)

    df_balanced = pd.concat(balanced, axis=0).sample(frac=1, random_state=42)

    df_balanced.to_csv(BALANCED_CSV, index=False)

    print(f"[INFO] Balanced CSV saved at: {BALANCED_CSV}")
    print(df_balanced['stage'].value_counts())


if __name__ == "__main__":
    main()
