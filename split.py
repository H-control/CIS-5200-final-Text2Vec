"""Split development JSONL into short/medium/long buckets by document token counts.

Produces:
 - data/nq_dev_short.jsonl
 - data/nq_dev_medium.jsonl
 - data/nq_dev_long.jsonl

Behavior:
 - short: doc_len < 33rd percentile
 - long:  doc_len > 66th percentile
 - medium: otherwise
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


DEV_PATH = Path("data/v1.0-simplified_nq-dev-all.jsonl")
OUT_SHORT = Path("data/nq_dev_short.jsonl")
OUT_MEDIUM = Path("data/nq_dev_medium.jsonl")
OUT_LONG = Path("data/nq_dev_long.jsonl")


def load_dev_lengths(dev_path: Path) -> List[int]:
    lengths = []
    with dev_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            lengths.append(len(item.get("document_tokens", [])))
    return lengths


def compute_thresholds(lengths: List[int]) -> Tuple[int, int]:
    arr = np.array(lengths)
    q1 = int(np.percentile(arr, 33))
    q2 = int(np.percentile(arr, 66))
    return q1, q2


def split_items(dev_path: Path, short_threshold: int, long_threshold: int) -> Tuple[List[dict], List[dict], List[dict]]:
    short_items, medium_items, long_items = [], [], []
    with dev_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            doc_len = len(item.get("document_tokens", []))
            if doc_len < short_threshold:
                short_items.append(item)
            elif doc_len > long_threshold:
                long_items.append(item)
            else:
                medium_items.append(item)
    return short_items, medium_items, long_items


def write_jsonl(items: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    if not DEV_PATH.exists():
        raise FileNotFoundError(f"Dev file not found: {DEV_PATH}")

    lengths = load_dev_lengths(DEV_PATH)
    if not lengths:
        raise RuntimeError("No items found in dev file")

    q1, q2 = compute_thresholds(lengths)
    print(f"Computed thresholds: short < {q1}, long > {q2}")

    short_items, medium_items, long_items = split_items(DEV_PATH, q1, q2)

    write_jsonl(short_items, OUT_SHORT)
    write_jsonl(medium_items, OUT_MEDIUM)
    write_jsonl(long_items, OUT_LONG)

    print(f"Wrote: {OUT_SHORT} ({len(short_items)}), {OUT_MEDIUM} ({len(medium_items)}), {OUT_LONG} ({len(long_items)})")


if __name__ == "__main__":
    main()

