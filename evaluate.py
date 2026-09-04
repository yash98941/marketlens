"""Scores the retrieval setups against each other on a public benchmark.

    python evaluate.py --mode bm25          fast, no model downloads
    python evaluate.py --mode ablation      the full table
    python evaluate.py --mode chunking      chunk size sweep

The numbers that matter:

  nDCG@10   the headline BEIR metric. Rewards putting relevant passages high,
            with a discount that falls off down the list.
  Recall@k  did the right passage make it into the top k at all. This is the
            ceiling for anything downstream, because a generator cannot use a
            passage that was never retrieved.
  MRR@10    one over the rank of the first relevant hit.
  latency   p50 and p95 per query. p95 is here because a mean hides the slow
            tail, and the tail is what a user notices.
"""

import argparse
import json
import os

import beir
from metrics import evaluate
from retrieve import BM25, Dense, Hybrid, Reranker


def show(rows):
    print("\n{:26s} {:>8s} {:>10s} {:>10s} {:>8s} {:>9s} {:>9s}".format(
        "method", "nDCG@10", "Recall@10", "Recall@50", "MRR@10", "p50 ms", "p95 ms"))
    print("-" * 88)
    for row in rows:
        print("{:26s} {:8.4f} {:10.4f} {:10.4f} {:8.4f} {:9.1f} {:9.1f}".format(
            row["method"], row["ndcg@10"], row["recall@10"], row["recall@50"],
            row["mrr@10"], row["latency_p50_ms"], row["latency_p95_ms"]))


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="fiqa")
parser.add_argument("--mode", default="bm25",
                    choices=["bm25", "ablation", "chunking"])
parser.add_argument("--limit", type=int, default=None,
                    help="only run this many queries, for a quick check")
parser.add_argument("--out", default="results")
args = parser.parse_args()

print("loading", args.dataset)
corpus, queries, qrels = beir.load(args.dataset)
doc_ids = sorted(corpus)
docs = [corpus[i] for i in doc_ids]
print("{} passages, {} queries with judgements".format(len(docs), len(queries)))

rows = []

if args.mode == "bm25":
    # Runs on a laptop with no model downloads, so it is the check that the
    # indexing and the metrics are right before spending time on embeddings.
    index = BM25().fit(docs)
    result = evaluate(index, doc_ids, queries, qrels, limit=args.limit)
    result["method"] = "bm25"
    rows.append(result)

elif args.mode == "ablation":
    # Built once and handed to every setup that needs them. Embedding 57k
    # passages takes minutes and none of the setups change the index.
    print("building the shared indexes")
    shared_bm25 = BM25().fit(docs)
    shared_dense = Dense().fit(docs)
    shared_reranker = Reranker()

    setups = [
        ("bm25 only", dict(use_bm25=True, use_dense=False, rerank=False)),
        ("dense only", dict(use_bm25=False, use_dense=True, rerank=False)),
        ("hybrid rrf", dict(use_bm25=True, use_dense=True, rerank=False)),
        ("bm25 + rerank", dict(use_bm25=True, use_dense=False, rerank=True)),
        ("hybrid + rerank", dict(use_bm25=True, use_dense=True, rerank=True)),
    ]
    for label, options in setups:
        print("running", label)
        retriever = Hybrid(bm25=shared_bm25, dense=shared_dense,
                           reranker=shared_reranker, **options).fit(docs)
        result = evaluate(retriever, doc_ids, queries, qrels, limit=args.limit)
        result["method"] = label
        rows.append(result)
        show([result])

elif args.mode == "chunking":
    # The BEIR passages are already short, so this splits and rejoins them to
    # show what the chunk setting does rather than to beat the plain baseline.
    import chunk

    for size, overlap in [(100, 0), (100, 25), (200, 50), (400, 100)]:
        print("\nchunk size {} overlap {}".format(size, overlap))
        pieces = []
        piece_ids = []
        for doc_id, text in zip(doc_ids, docs):
            for part in chunk.chunk_text(text, size, overlap):
                pieces.append(part)
                piece_ids.append(doc_id)

        index = BM25().fit(pieces)
        result = evaluate(index, piece_ids, queries, qrels, limit=args.limit)
        result["method"] = "size {} overlap {}".format(size, overlap)
        result["chunks"] = len(pieces)
        rows.append(result)

show(rows)

os.makedirs(args.out, exist_ok=True)
path = os.path.join(args.out, "{}_{}.json".format(args.dataset, args.mode))
with open(path, "w") as f:
    json.dump({"dataset": args.dataset, "mode": args.mode,
               "passages": len(docs), "rows": rows}, f, indent=2)
print("\nSaved", path)
