import json
from pathlib import Path
from rapidfuzz import fuzz
import pandas as pd

PRED_PATH = Path("outputs/cadec_llm_predictions.jsonl")
OUT_PATH = Path("outputs/cadec_eval_summary.csv")

def normalize(x):
    return str(x).lower().strip()

def get_pred_mentions(parsed):
    events = parsed.get("adverse_events", [])
    mentions = []
    terms = []
    for ev in events:
        if isinstance(ev, dict):
            if ev.get("mention"):
                mentions.append(normalize(ev["mention"]))
            if ev.get("normalized_term"):
                terms.append(normalize(ev["normalized_term"]))
        elif isinstance(ev, str):
            mentions.append(normalize(ev))
    return list(set(mentions)), list(set(terms))

def soft_match(a, b, threshold=85):
    return fuzz.partial_ratio(a, b) >= threshold or fuzz.token_set_ratio(a, b) >= threshold

def score_instance(gold, pred, threshold=85):
    gold = [normalize(x) for x in gold if x]
    pred = [normalize(x) for x in pred if x]

    matched_gold = set()
    matched_pred = set()

    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            if j in matched_pred:
                continue
            if soft_match(g, p, threshold=threshold):
                matched_gold.add(i)
                matched_pred.add(j)
                break

    tp = len(matched_gold)
    fp = len(pred) - tp
    fn = len(gold) - tp

    return tp, fp, fn

def main():
    rows = [json.loads(line) for line in open(PRED_PATH)]
    records = []

    total_tp = total_fp = total_fn = 0

    for r in rows:
        pred_mentions, pred_terms = get_pred_mentions(r["parsed_output"])

        # Mention-level score
        tp, fp, fn = score_instance(r["gold_ades"], pred_mentions)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        records.append({
            "idx": r["idx"],
            "n_gold": len(r["gold_ades"]),
            "n_pred": len(pred_mentions),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gold_ades": r["gold_ades"],
            "pred_mentions": pred_mentions,
            "text": r["text"],
        })

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)

    print({
        "n_instances": len(rows),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    })

    print(f"Saved per-instance eval to {OUT_PATH}")

if __name__ == "__main__":
    main()