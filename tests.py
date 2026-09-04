"""Checks for the parts that fail quietly.

Run with: python tests.py

Nothing here downloads a model or hits the network, so it runs in a couple of
seconds. The metric tests use examples small enough to work out by hand, which
is the only way to be sure a scoring bug is not quietly flattering the results.
"""

import chunk
import generate
import metrics
from fetch import check_url
from retrieve import BM25, fuse, tokenise

DOCS = [
    "The central bank raised interest rates by 50 basis points today.",
    "Quarterly revenue grew 12 percent while operating margin fell.",
    "Interest rates were left unchanged at the previous meeting.",
    "The company announced a share buyback worth two billion dollars.",
]


def test_tokenise_drops_stopwords():
    tokens = tokenise("What is the revenue of the company?")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "revenue" in tokens and "company" in tokens


def test_bm25_finds_the_obvious_match():
    index = BM25().fit(DOCS)
    hits = index.search("share buyback", k=2)
    assert hits, "no results at all"
    assert hits[0][0] == 3, "best hit should be the buyback sentence"


def test_bm25_returns_nothing_for_unknown_words():
    index = BM25().fit(DOCS)
    assert index.search("zebra kangaroo", k=3) == []


def test_bm25_ranks_by_idf_not_just_count():
    """A word in every document should count for almost nothing.

    Without the idf term, a query made of common words would return whichever
    document repeats them most, which is the classic broken keyword search.
    """
    docs = ["market market market news", "market rare_term news"]
    index = BM25().fit(docs)
    hits = dict(index.search("market rare_term", k=2))
    assert hits[1] > hits[0], "the rare word should outweigh three common ones"


def test_bm25_penalises_padded_documents():
    """Length normalisation, the b term.

    Both documents mention the query word once. The padded one should not win
    just because it is longer.
    """
    docs = ["dividend announced", "dividend announced " + "filler " * 200]
    index = BM25().fit(docs)
    hits = dict(index.search("dividend", k=2))
    assert hits[0] > hits[1], "the padded document scored higher"


def test_chunking_keeps_all_the_words():
    text = " ".join("sentence number {}.".format(i) for i in range(60))
    pieces = chunk.chunk_text(text, size=50, overlap=0)
    assert len(pieces) > 1, "text was not split at all"
    rejoined = " ".join(pieces).split()
    assert len(rejoined) == len(text.split()), "words were lost or duplicated"


def test_chunk_overlap_repeats_the_boundary():
    text = " ".join("word{} is here.".format(i) for i in range(80))
    plain = chunk.chunk_text(text, size=40, overlap=0)
    lapped = chunk.chunk_text(text, size=40, overlap=15)
    assert len(lapped) > len(plain), "overlap should produce more chunks"


def test_chunk_rejects_silly_overlap():
    try:
        chunk.chunk_text("a. b. c.", size=10, overlap=10)
    except ValueError:
        return
    assert False, "overlap equal to size should have been rejected"


def test_chunk_documents_adds_the_title():
    docs = [{"url": "http://x.test/a", "title": "Acme Q3",
             "text": "The company said margins improved. Revenue was flat."}]
    texts, sources = chunk.chunk_documents(docs, size=100, overlap=0)
    assert texts[0].startswith("Acme Q3."), "title was not prefixed"
    assert sources[0]["url"] == "http://x.test/a"
    assert len(texts) == len(sources)


def test_fusion_rewards_agreement():
    """A document both retrievers found should beat one only seen by a single
    retriever, even when that single one ranked it first."""
    bm25 = [(1, 9.0), (2, 8.0), (3, 7.0)]
    dense = [(4, 0.9), (2, 0.8), (5, 0.7)]
    merged = fuse([bm25, dense], k=3)
    assert merged[0][0] == 2, "the agreed document should come first"


def test_ndcg_is_one_for_a_perfect_ranking():
    relevant = {"a": 1, "b": 1}
    assert abs(metrics.ndcg_at_k(["a", "b", "c"], relevant, 10) - 1.0) < 1e-9


def test_ndcg_drops_when_the_hit_moves_down():
    relevant = {"a": 1}
    high = metrics.ndcg_at_k(["a", "x", "y"], relevant, 10)
    low = metrics.ndcg_at_k(["x", "y", "a"], relevant, 10)
    assert high > low
    # Rank 3 means a discount of 1/log2(4) = 0.5 exactly.
    assert abs(low - 0.5) < 1e-9


def test_ndcg_ignores_hits_past_the_cutoff():
    relevant = {"a": 1}
    assert metrics.ndcg_at_k(["x", "y", "z", "a"], relevant, 3) == 0.0


def test_recall_counts_only_the_top_k():
    relevant = {"a": 1, "b": 1, "c": 1}
    assert abs(metrics.recall_at_k(["a", "b", "z"], relevant, 3) - 2 / 3) < 1e-9
    assert abs(metrics.recall_at_k(["a", "b", "z"], relevant, 1) - 1 / 3) < 1e-9


def test_mrr_uses_the_first_hit_only():
    relevant = {"b": 1, "c": 1}
    assert abs(metrics.mrr_at_k(["a", "b", "c"], relevant, 10) - 0.5) < 1e-9
    assert metrics.mrr_at_k(["x", "y"], relevant, 10) == 0.0


def test_url_check_blocks_private_targets():
    """The server side request forgery guard.

    Each of these would let someone use the server to reach something they
    cannot reach themselves.
    """
    bad = [
        "http://127.0.0.1:8000/admin",
        "http://localhost/secret",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://[::1]/",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com",
        "not-a-url",
    ]
    for url in bad:
        try:
            check_url(url)
        except ValueError:
            continue
        assert False, "should have refused: " + url


def test_url_check_allows_a_normal_site():
    parsed = check_url("https://example.com/news/story")
    assert parsed.hostname == "example.com"


def test_extractive_answer_picks_the_right_sentence():
    passages = [
        "The weather was mild. Operating margin fell to 14 percent this quarter.",
        "Unrelated text about shipping schedules and warehouses.",
    ]
    out = generate.extractive_answer("What happened to operating margin?",
                                     passages, max_sentences=1)
    assert "14 percent" in out


def test_extractive_answer_admits_when_it_has_nothing():
    out = generate.extractive_answer("zebra kangaroo trampoline",
                                     ["Interest rates were unchanged."])
    assert "Nothing" in out or "answers that" in out


passed = 0
failed = 0
for name, item in sorted(globals().items()):
    if not name.startswith("test_"):
        continue
    try:
        item()
        print("pass  " + name)
        passed += 1
    except AssertionError as err:
        print("FAIL  {}: {}".format(name, err))
        failed += 1

print("\n{} passed, {} failed".format(passed, failed))
