import pandas as pd
from pathlib import Path

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def normalize_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def build_doc_level(split: str):
    df = pd.read_csv(DATA / f"cadec_{split}.csv")

    df["text_norm"] = df["text"].apply(normalize_str)
    df["ade_norm"] = df["ade"].apply(normalize_str)
    df["term_norm"] = df["term_PT"].apply(normalize_str)

    grouped = (
        df.groupby("text_norm")
        .agg(
            text=("text", "first"),
            gold_ades=("ade_norm", lambda x: sorted(set([v for v in x if v]))),
            gold_terms=("term_norm", lambda x: sorted(set([v for v in x if v]))),
        )
        .reset_index(drop=True)
    )

    grouped["n_gold_ades"] = grouped["gold_ades"].apply(len)
    grouped["n_gold_terms"] = grouped["gold_terms"].apply(len)

    grouped.to_json(OUT / f"cadec_{split}_doclevel.jsonl", orient="records", lines=True)
    grouped.to_csv(OUT / f"cadec_{split}_doclevel.csv", index=False)

    print(split, grouped.shape)
    print(grouped[["text", "gold_ades", "gold_terms"]].head())

if __name__ == "__main__":
    build_doc_level("train")
    build_doc_level("test")