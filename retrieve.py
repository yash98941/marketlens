"""Retrieval methods, from a plain keyword index up to a reranked hybrid.

There are four things here and they stack:

  BM25        keyword scoring, written out rather than imported so the
              saturation and length terms are visible
  Dense       sentence embeddings with cosine similarity over a FAISS index
  Hybrid      reciprocal rank fusion of the two lists above
  Rerank      a cross encoder rescores the top candidates

Keyword search and embedding search fail in different ways. BM25 misses a
question that uses none of the words in the answer. Dense search drifts to
passages that are about roughly the right topic but do not contain the number
you asked for. Fusing the two rankings covers more than either alone, and the
cross encoder then fixes the ordering of whatever survived.
"""

import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "what", "how", "why", "do", "does", "i", "you",
}


def tokenise(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25.

    Score of a document for a query is a sum over query terms of

        idf(term) * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avg_length))

    where f is how often the term shows up in that document. The k1 part is
    saturation: the tenth mention of a word is worth much less than the second.
    The b part is length normalisation, so a long document does not win just by
    being long enough to contain everything.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avg_len = 0.0
        self.postings = {}
        self.idf = {}
        self.n_docs = 0

    def fit(self, docs):
        """docs is a list of strings, one per chunk."""
        self.n_docs = len(docs)
        self.postings = {}
        self.doc_len = []

        for i, text in enumerate(docs):
            terms = tokenise(text)
            self.doc_len.append(len(terms))
            for term, count in Counter(terms).items():
                self.postings.setdefault(term, []).append((i, count))

        self.doc_len = np.array(self.doc_len, dtype=float)
        self.avg_len = self.doc_len.mean() if self.n_docs else 0.0

        # Smoothed idf. The +0.5s and the +1 keep it positive for terms that
        # appear in nearly every document, which the textbook version does not.
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

        return self

    def search(self, query, k=10):
        scores = np.zeros(self.n_docs)

        for term in tokenise(query):
            posting = self.postings.get(term)
            if posting is None:
                continue

            idf = self.idf[term]
            for doc_id, freq in posting:
                norm = 1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len
                scores[doc_id] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)

        return top_k(scores, k)


def top_k(scores, k):
    """Indices of the k highest scores, best first, ignoring zeros."""
    k = min(k, len(scores))
    part = np.argpartition(scores, -k)[-k:]
    order = part[np.argsort(scores[part])[::-1]]
    return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


class Dense:
    """Embeddings in a FAISS index.

    Vectors are normalised so inner product is cosine similarity, which lets
    the flat inner product index stand in for exact cosine search.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.index = None

    def load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def encode(self, texts, query=False):
        model = self.load_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=not query and len(texts) > 500,
        )
        return vectors.astype("float32")

    def fit(self, docs):
        import faiss

        vectors = self.encode(docs)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self

    def search(self, query, k=10):
        vector = self.encode([query], query=True)
        scores, ids = self.index.search(vector, min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def fuse(rankings, k=10, constant=60):
    """Reciprocal rank fusion.

    Each list votes with 1 / (constant + rank). Using rank instead of score
    means BM25 scores and cosine similarities never have to be put on the same
    scale, which is the part that usually goes wrong when people combine them.
    The constant stops the top hit of one list from steamrolling the other.
    """
    totals = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            totals[doc_id] = totals.get(doc_id, 0.0) + 1.0 / (constant + rank + 1)

    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [(doc_id, score) for doc_id, score in ordered[:k]]


class Reranker:
    """Cross encoder over query and passage together.

    The retrievers above embed the query and the passage separately, so they
    never get to compare the words in one against the words in the other. A
    cross encoder reads both at once, which is far more accurate and far too
    slow to run over the whole corpus. So it only ever sees the shortlist.
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def load_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        return self.model

    def rerank(self, query, candidates, docs, k=10):
        if not candidates:
            return []

        model = self.load_model()
        ids = [doc_id for doc_id, _ in candidates]
        pairs = [(query, docs[i]) for i in ids]
        scores = model.predict(pairs, batch_size=self.batch_size)

        order = np.argsort(scores)[::-1][:k]
        return [(ids[i], float(scores[i])) for i in order]


class Hybrid:
    """The full pipeline, with each stage able to be switched off.

    Keeping the switches here is what makes the ablation table in evaluate.py
    a few lines instead of four near copies of the same code.

    The three index arguments let several setups share one built index.
    Embedding a corpus takes minutes, and the ablation compares five setups
    over the same documents, so building it once and passing it around saves
    most of the run.
    """

    def __init__(self, use_bm25=True, use_dense=True, rerank=False,
                 bm25=None, dense=None, reranker=None, candidates=50):
        self.use_bm25 = use_bm25
        self.use_dense = use_dense
        self.candidates = candidates

        self.bm25 = (bm25 or BM25()) if use_bm25 else None
        self.dense = (dense or Dense()) if use_dense else None
        self.reranker = (reranker or Reranker()) if rerank else None
        self.docs = []

    def fit(self, docs):
        self.docs = docs
        # A shared index arrives already built, so do not pay for it twice.
        if self.bm25 and self.bm25.n_docs == 0:
            self.bm25.fit(docs)
        if self.dense and self.dense.index is None:
            self.dense.fit(docs)
        return self

    def search(self, query, k=10):
        # Pull a wider shortlist than asked for when something downstream is
        # going to reorder it, otherwise the reranker has nothing to work with.
        wide = self.candidates if self.reranker else k

        rankings = []
        if self.bm25:
            rankings.append(self.bm25.search(query, wide))
        if self.dense:
            rankings.append(self.dense.search(query, wide))

        if not rankings:
            raise ValueError("turn on at least one of bm25 or dense")

        merged = rankings[0] if len(rankings) == 1 else fuse(rankings, wide)

        if self.reranker:
            return self.reranker.rerank(query, merged, self.docs, k)
        return merged[:k]

    def name(self):
        parts = []
        if self.use_bm25:
            parts.append("bm25")
        if self.use_dense:
            parts.append("dense")
        label = " + ".join(parts)
        if self.reranker:
            label += " + rerank"
        return label
