import asyncio
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import sys

import pandas as pd

# Ensure project root on path to import generate_submission_csv
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_submission_csv import generate_predictions

TRAIN_CSV = Path("data/train.csv")
OUTPUT_SUBMISSION = Path("submission_eval.csv")
K = 10


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """
    Load train.csv and aggregate ground-truth URLs per query.
    Returns: mapping query -> set(urls)
    """
    gt: Dict[str, Set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("Query", "").strip()
            u = row.get("Assessment_url", "").strip()
            if q and u:
                gt[q].add(u)
    return gt


def load_predictions(path: Path, k: int) -> Dict[str, List[str]]:
    """
    Load submission CSV and take top-k predicted URLs per query.
    Returns: mapping query -> list(urls) ordered
    """
    pred_df = pd.read_csv(path)
    grouped: Dict[str, List[str]] = defaultdict(list)
    for _, row in pred_df.iterrows():
        q = str(row.get("Query", "")).strip()
        u = str(row.get("Assessment_url", "")).strip()
        if not q:
            continue
        if len(grouped[q]) < k:
            grouped[q].append(u)
    return grouped


def recall_at_k(gt_urls: Set[str], pred_urls: List[str], k: int) -> float:
    if not gt_urls:
        return 0.0
    topk = pred_urls[:k]
    hits = sum(1 for u in gt_urls if u in topk)
    return hits / len(gt_urls)


def evaluate(gt: Dict[str, Set[str]], preds: Dict[str, List[str]], k: int) -> Tuple[float, int]:
    recalls: List[float] = []
    for q, urls in gt.items():
        r = recall_at_k(urls, preds.get(q, []), k)
        recalls.append(r)
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    return mean_recall, len(recalls)


async def main():
    # 1) Generate predictions for train.csv
    await generate_predictions(
        test_csv_path=str(TRAIN_CSV),
        output_path=str(OUTPUT_SUBMISSION),
        top_k=K,
    )

    # 2) Load ground-truth and predictions
    gt = load_ground_truth(TRAIN_CSV)
    preds = load_predictions(OUTPUT_SUBMISSION, K)

    # 3) Compute Recall@K
    mean_recall, n = evaluate(gt, preds, K)

    print("=== Evaluation Results ===")
    print(f"Queries evaluated: {n}")
    print(f"Recall@{K}: {mean_recall:.4f}")
    print(f"Submission file: {OUTPUT_SUBMISSION}")


if __name__ == "__main__":
    asyncio.run(main())

