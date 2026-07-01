import os
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_PATH = Path("outputs/cadec_test_doclevel.jsonl")
PROMPT_PATH = Path("prompts/ade_extraction_prompt.txt")
OUT_PATH = Path("outputs/cadec_llm_predictions.jsonl")

def load_jsonl(path):
    return [json.loads(line) for line in open(path)]

def safe_json_parse(s):
    try:
        return json.loads(s)
    except Exception:
        # simple cleanup fallback
        s = s.strip()
        if s.startswith("```"):
            s = s.strip("`")
            s = s.replace("json", "", 1).strip()
        try:
            return json.loads(s)
        except Exception:
            return {"adverse_events": [], "parse_error": s}

def call_llm(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content

def main(n=25):
    rows = load_jsonl(DATA_PATH)
    prompt_template = PROMPT_PATH.read_text()

    existing = set()
    if OUT_PATH.exists():
        for line in open(OUT_PATH):
            try:
                existing.add(json.loads(line)["idx"])
            except Exception:
                pass

    with open(OUT_PATH, "a") as f:
        for idx, row in tqdm(list(enumerate(rows[:n]))):
            if idx in existing:
                continue

            prompt = prompt_template.replace("{text}", row["text"])

            try:
                raw = call_llm(prompt)
                parsed = safe_json_parse(raw)
            except Exception as e:
                raw = ""
                parsed = {"adverse_events": [], "error": str(e)}

            record = {
                "idx": idx,
                "text": row["text"],
                "gold_ades": row["gold_ades"],
                "gold_terms": row["gold_terms"],
                "raw_output": raw,
                "parsed_output": parsed,
            }

            f.write(json.dumps(record) + "\n")
            f.flush()
            time.sleep(0.2)

if __name__ == "__main__":
    main(n=25)