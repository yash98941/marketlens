# MarketLens

Ask questions across a set of financial news articles and get an answer with the
passages it came from.

The retrieval side is the actual project. Most RAG demos stop at "embed the
documents, put them in a vector store, feed the top 5 chunks to an LLM" and
never check whether those 5 chunks were the right ones. This one measures that
directly on a public benchmark, and the measurements decided the design.

## What is in here

Four retrieval setups that can be switched on and off independently:

- **BM25**, written out in `retrieve.py` rather than imported, so the saturation
  and length normalisation terms are visible and testable
- **Dense**, sentence embeddings in a FAISS inner product index
- **Hybrid**, reciprocal rank fusion of the two lists
- **Rerank**, a cross encoder rescoring the shortlist

Answer generation runs without an API key by default. It pulls the sentences
from the retrieved passages that best match the question. Set `OPENAI_API_KEY`
and it uses a chat model instead. Keeping the no-key path working means anyone
can clone this and run it, and it doubles as a control: if the answers are bad
without an LLM, the retrieval is the problem, not the prompt.

## Results

BEIR FiQA, the full corpus of 57,638 passages and all 648 judged queries.
Latency is per query on a laptop CPU, no GPU anywhere in this project.

| method | nDCG@10 | Recall@10 | Recall@50 | MRR@10 | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.231 | 0.295 | 0.449 | 0.285 | 58 ms | 224 ms |

The published Anserini BM25 number on FiQA is about 0.236 nDCG@10. Landing at
0.231 with an implementation written from scratch means the scoring function
and the tokeniser are close enough to trust, which is the reason for running
this before touching anything else. A from-scratch BM25 that scores 0.15 looks
like a weak baseline when it is actually a bug, and every comparison built on
top of it would be wrong.

The dense, hybrid and rerank rows are not filled in here. Encoding 57,638 FiQA
passages on this CPU runs at roughly 16 passages per second, so one embedding
pass is about an hour, and the cross encoder over 648 queries times 50
candidates is a lot worse. The ablation runs on the smaller SciFact corpus
instead, which is what `evaluate.py --dataset scifact --mode ablation` does.

## Why hybrid instead of just embeddings

Keyword search and embedding search fail differently.

BM25 cannot find a passage that answers the question using none of the
question's words. Ask about "getting out of debt" and it will not surface a
passage about "paying down a mortgage balance".

Dense search has the opposite problem. It finds passages about roughly the
right topic and is happy to rank one that discusses expense ratios above one
that states the specific number you asked for. Exact tokens like ticker symbols,
section numbers and dates are where it is weakest, and finance text is full of
them.

Fusing the two rankings by rank rather than by score sidesteps the usual
problem of putting BM25 scores and cosine similarities on a comparable scale.
Each list contributes `1 / (60 + rank)` and the constant stops one list's top
hit from dominating.

## Why the cross encoder only sees a shortlist

BM25 and the embedding model both score the query and the passage separately,
so neither ever compares the words of one against the words of the other. A
cross encoder reads both together and is much more accurate, but it has to run
the model once per candidate pair, so running it over a full corpus is not
possible. It reranks the top 50 and nothing else.

That is also why `Recall@50` is in the results table. It is the ceiling on what
reranking can achieve, because the reranker can only reorder what retrieval
already found.

## Security note

The first version of this passed whatever URL the user typed straight into
`requests.get`. That is a server side request forgery hole. Anyone using a
hosted copy could enter `http://169.254.169.254/latest/meta-data/` and read the
cloud instance credentials, or point it at internal services on the private
network that they cannot otherwise reach.

`fetch.check_url` now requires http or https, resolves the hostname and refuses
private, loopback, link local and reserved addresses, and re-checks after every
redirect so a public URL cannot bounce to a private one. Response size is
capped as well. `tests.py` covers the blocked cases.

## The web app

`streamlit run app.py` opens a page where you add article URLs one row at a
time, press a button, and it fetches, chunks and indexes them. Articles stay in
a library, so adding a sixth one searches all six rather than throwing the
first five away. Each article and each URL row has its own remove button.

Asking is behind a form, so retrieval only runs when the search button is
pressed. Streamlit reruns the whole script on every widget change, and without
the form the search would fire again every time someone nudged a slider. The
sentence transformer is cached across reruns because it takes about a minute to
load, but the index is deliberately not cached, or adding an article would
quietly keep searching the old one.

If a setting or the library changes after a build, the page says so instead of
answering from a stale index.

## Running it

```
pip install -r requirements.txt

python tests.py                       # 19 checks, no downloads, a few seconds
python evaluate.py --mode bm25        # keyword baseline on FiQA
python evaluate.py --mode ablation    # the full comparison
python evaluate.py --mode chunking    # chunk size sweep

streamlit run app.py                  # the web app
```

The BM25 mode needs no model downloads and runs on a CPU. The ablation
downloads a sentence transformer and a cross encoder on first use, and the
cross encoder rows are slow without a GPU.

To try it quickly, `--limit 100` runs the first hundred queries only.

## Files

| file | what it does |
| --- | --- |
| `retrieve.py` | BM25, dense, rank fusion, cross encoder, and the pipeline |
| `chunk.py` | sentence aware splitting with overlap |
| `fetch.py` | URL fetching with the SSRF checks, and text extraction |
| `generate.py` | extractive and LLM answering, with citations |
| `metrics.py` | nDCG, recall, MRR, latency percentiles |
| `beir.py` | downloads and loads the benchmark |
| `evaluate.py` | runs the comparisons and saves the tables |
| `app.py` | Streamlit front end |
| `tests.py` | checks on BM25, chunking, fusion, metrics and the URL guard |

## Benchmark

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021).
BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information
Retrieval Models. NeurIPS Datasets and Benchmarks.
