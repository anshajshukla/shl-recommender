"""
Compare finance-specific Recall@K between two evaluation JSON files.

Usage:
    python tools/check_finance_baseline.py baseline_eval.json new_eval.json --k 10 \
        --keywords finance,financial,accounting,analyst

The script tries to be tolerant of different evaluation JSON shapes. It expects that
each evaluation contains a list of per-query entries under one of these keys:
    - "per_query", "queries", "results", "items"

Each per-query entry should contain at least:
    - a `query` text field (string)
    - a list of ground-truth URLs under one of: `gold_urls`, `ground_truth`, `relevant_urls`, `answers`
    - a list of predicted URLs under one of: `predicted_urls`, `predictions`, `recommended`, `results`

If ground-truth is not URL-based but uses assessment ids/names, the script will still
attempt to compare by string equality.

Output: prints baseline, new, and delta for Finance Recall@K (mean over finance queries).
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

FINANCE_KEYWORDS_DEFAULT = ["finance", "financial", "accounting", "analyst", "bookkeeping", "valuation"]

# candidate keys for per-query lists and fields
PER_QUERY_KEYS = ["per_query", "queries", "results", "items"]
GOLD_KEYS = ["gold_urls", "ground_truth", "relevant_urls", "answers", "gold", "ground_truth_urls"]
PRED_KEYS = ["predicted_urls", "predictions", "recommended", "results", "predicted", "preds"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_per_query(obj: Dict) -> List[Dict]:
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        return []
    for k in PER_QUERY_KEYS:
        if k in obj and isinstance(obj[k], list):
            return obj[k]
    # fallback: look for first top-level list
    for v in obj.values():
        if isinstance(v, list):
            return v
    return []


def extract_field(item: Dict, candidates: List[str]):
    for k in candidates:
        if k in item:
            return item[k]
    # fallback: look for list-valued fields
    for k, v in item.items():
        if isinstance(v, list):
            return v
    return []


def normalize_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip().lower() for i in x if i is not None]
    # sometimes stored as comma-separated string
    if isinstance(x, str):
        return [part.strip().lower() for part in x.split(",") if part.strip()]
    return [str(x).lower()]


def is_finance_query(query_text: str, finance_keywords: List[str]) -> bool:
    q = (query_text or "").lower()
    return any(k in q for k in finance_keywords)


def recall_at_k_for_query(gold: List[str], pred: List[str], k: int) -> float:
    if not gold:
        return 0.0
    pred_topk = pred[:k]
    # count of gold items retrieved in top-k
    found = sum(1 for g in gold if g in pred_topk)
    # recall for this query: proportion of gold items found
    return found / len(gold)


def aggregate_recall_at_k(per_query: List[Dict], finance_keywords: List[str], k: int) -> Dict[str, Any]:
    finance_recalls = []
    count_finance_queries = 0

    for item in per_query:
        qtext = item.get("query") or item.get("text") or item.get("job") or ""
        if not isinstance(qtext, str):
            qtext = str(qtext)
        if not is_finance_query(qtext, finance_keywords):
            continue
        count_finance_queries += 1

        gold_raw = extract_field(item, GOLD_KEYS)
        pred_raw = extract_field(item, PRED_KEYS)

        gold = normalize_list(gold_raw)
        pred = normalize_list(pred_raw)

        r = recall_at_k_for_query(gold, pred, k)
        finance_recalls.append(r)

    mean_recall = sum(finance_recalls) / len(finance_recalls) if finance_recalls else 0.0
    return {
        "n_queries": count_finance_queries,
        "mean_recall_at_{}".format(k): mean_recall,
        "per_query_recalls": finance_recalls
    }


def main():
    parser = argparse.ArgumentParser(description="Compute finance-specific Recall@K between two eval JSONs")
    parser.add_argument("baseline", help="Path to baseline evaluation JSON")
    parser.add_argument("new", help="Path to new evaluation JSON")
    parser.add_argument("--k", type=int, default=10, help="Top-K for recall")
    parser.add_argument("--keywords", type=str, default=",".join(FINANCE_KEYWORDS_DEFAULT),
                        help="Comma-separated finance keywords to filter queries")

    args = parser.parse_args()
    baseline_path = Path(args.baseline)
    new_path = Path(args.new)
    k = args.k
    finance_keywords = [kw.strip().lower() for kw in args.keywords.split(",") if kw.strip()]

    if not baseline_path.exists():
        print(f"Baseline file not found: {baseline_path}")
        sys.exit(2)
    if not new_path.exists():
        print(f"New eval file not found: {new_path}")
        sys.exit(2)

    b = load_json(baseline_path)
    n = load_json(new_path)

    per_b = find_per_query(b)
    per_n = find_per_query(n)

    if not per_b:
        print(f"Could not find per-query list in baseline evaluation ({baseline_path}).")
    if not per_n:
        print(f"Could not find per-query list in new evaluation ({new_path}).")

    agg_b = aggregate_recall_at_k(per_b, finance_keywords, k)
    agg_n = aggregate_recall_at_k(per_n, finance_keywords, k)

    mb = agg_b.get(f"mean_recall_at_{k}", 0.0)
    mn = agg_n.get(f"mean_recall_at_{k}", 0.0)
    nb = agg_b.get("n_queries", 0)
    nn = agg_n.get("n_queries", 0)

    print("Finance-specific Recall@{}:".format(k))
    print(f"  baseline: {mb:.4f}  (n_queries={nb})")
    print(f"  new:      {mn:.4f}  (n_queries={nn})")
    print(f"  delta:    {mn - mb:+.4f}")

    # Exit code 0 for success

if __name__ == "__main__":
    main()
