import json
from tqdm import tqdm


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def is_valid_gold(item):
    """
    Return True if gold_start, gold_end are valid.
    """
    annotations = item.get("annotations", [])
    if not annotations:
        return False
    
    ann = annotations[0]

    # Case 1: short answer
    if ann.get("short_answers"):
        start = ann["short_answers"][0].get("start_token", -1)
        end = ann["short_answers"][0].get("end_token", -1)

    # Case 2: long answer
    else:
        la = ann.get("long_answer", {})
        start = la.get("start_token", -1)
        end = la.get("end_token", -1)

    # validity check
    if start is None or end is None:
        return False
    if start < 0 or end < 0:
        return False
    if start == end:
        return False  # empty answer
    return True


def clean_dataset(input_path, output_path):
    total = 0
    kept = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as fw:
        for item in tqdm(load_jsonl(input_path), desc=f"Cleaning {input_path}"):
            total += 1
            if is_valid_gold(item):
                fw.write(json.dumps(item) + "\n")
                kept += 1
            else:
                skipped += 1

    print(f"\n=== Cleaning Summary for {input_path} ===")
    print(f"Total samples: {total}")
    print(f"Kept valid samples: {kept}")
    print(f"Skipped invalid samples: {skipped}")
    print(f"Saved cleaned dataset to: {output_path}\n")


# ---------- Process your 3 datasets ----------
dataset = "data/v1.0-simplified_nq-dev-all.jsonl"

clean_dataset(dataset, "data/v1.0-simplified_nq-dev-all_cleaned.jsonl")