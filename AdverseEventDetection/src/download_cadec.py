from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT = Path("data")
OUT.mkdir(exist_ok=True)

dataset = load_dataset("KevinSpaghetti/cadec")

for split in dataset:
    df = dataset[split].to_pandas()
    df.to_csv(OUT / f"cadec_{split}.csv", index=False)
    print(split, df.shape)
    print(df.head())

print(dataset)