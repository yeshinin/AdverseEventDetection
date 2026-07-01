#found a common row between train and test set

import pandas as pd
from pathlib import Path

DATA = Path("data")

LEAK_TEXT = "muscle pain."
LEAK_ADE = "muscle pain"
LEAK_TERM = "myalgia"

def remove_exact_row(split: str):
    path = DATA / f"cadec_{split}.csv"
    df = pd.read_csv(path)

    mask = (
        (df["text"].astype(str).str.strip() == LEAK_TEXT) &
        (df["ade"].astype(str).str.strip() == LEAK_ADE) &
        (df["term_PT"].astype(str).str.strip() == LEAK_TERM)
    )

    print(f"{split}: found {mask.sum()} leaked rows")

    cleaned = df.loc[~mask].copy()

    backup_path = DATA / f"cadec_{split}_original_with_leak.csv"
    cleaned_path = DATA / f"cadec_{split}.csv"

    # backup original before overwriting
    df.to_csv(backup_path, index=False)
    cleaned.to_csv(cleaned_path, index=False)

    print(f"{split}: before={len(df)}, after={len(cleaned)}")
    print(f"Backup saved to: {backup_path}")

if __name__ == "__main__":
    # Recommended: remove from train only, keep test unchanged
    remove_exact_row("train")