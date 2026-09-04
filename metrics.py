"""Retrieval metrics, kept apart from the script that runs them so they can be
imported and tested on hand worked examples.

  nDCG@k    graded, position weighted. The standard BEIR headline number.
  Recall@k  fraction of the relevant passages that made the top k.
  MRR@k     one over the rank of the first relevant hit.
"""

import time

import numpy as np


def dcg(grades):
    return sum(g / np.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked, relevant, k=10):
    """Divided by the best ordering possible, so 1.0 means perfect."""
    grades = [relevant.get(doc_id, 0) for doc_id in ranked[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(grades) / best if best > 0 else 0.0


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def mrr_at_k(ranked, relevant, k=10):
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(retriever, doc_ids, queries, qrels, limit=None):
    """Runs every query, averages the metrics and records how long each took."""
    items = sorted(queries.items())
    if limit:
        items = items[:limit]

    scores = {"ndcg@10": [], "recall@10": [], "recall@50": [], "mrr@10": []}
    times = []

    for qid, text in items:
        start = time.perf_counter()
        hits = retriever.search(text, k=50)
        times.append((time.perf_counter() - start) * 1000)

        ranked = [doc_ids[i] for i, _ in hits]
        relevant = qrels[qid]

        scores["ndcg@10"].append(ndcg_at_k(ranked, relevant, 10))
        scores["recall@10"].append(recall_at_k(ranked, relevant, 10))
        scores["recall@50"].append(recall_at_k(ranked, relevant, 50))
        scores["mrr@10"].append(mrr_at_k(ranked, relevant, 10))

    out = {name: float(np.mean(values)) for name, values in scores.items()}
    out["queries"] = len(items)
    out["latency_p50_ms"] = float(np.percentile(times, 50))
    out["latency_p95_ms"] = float(np.percentile(times, 95))
    return out
