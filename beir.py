"""Downloads and loads a BEIR dataset, used here for the retrieval numbers.

FiQA is financial question answering: real questions from investor forums
against a corpus of finance passages, with human judgements of which passage
answers which question. It matches what this project is for, and because it is
a public benchmark the scores can be compared against published ones instead
of being marked by me.

Files after unzipping:
    corpus.jsonl     one passage per line, with _id, title, text
    queries.jsonl    one question per line, with _id, text
    qrels/test.tsv   query-id, corpus-id, relevance
"""

import json
import os
import zipfile

import requests

URLS = {
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
}

CACHE = "data"


def download(name):
    """Returns the folder for this dataset, downloading it the first time."""
    if name not in URLS:
        raise ValueError("no url for dataset: " + name)

    folder = os.path.join(CACHE, name)
    if os.path.isdir(folder):
        return folder

    os.makedirs(CACHE, exist_ok=True)
    archive = os.path.join(CACHE, name + ".zip")

    if not os.path.exists(archive):
        print("downloading", name, "...")
        response = requests.get(URLS[name], stream=True, timeout=120)
        response.raise_for_status()
        with open(archive, "wb") as f:
            for block in response.iter_content(chunk_size=1 << 20):
                f.write(block)

    print("unzipping", name, "...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(CACHE)

    return folder


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load(name="fiqa", split="test"):
    """Returns corpus, queries and qrels.

    corpus  maps doc id to the passage text, title glued on the front
    queries maps query id to the question, only those that have judgements
    qrels   maps query id to a dict of doc id to relevance grade
    """
    folder = download(name)

    corpus = {}
    for row in read_jsonl(os.path.join(folder, "corpus.jsonl")):
        title = row.get("title", "").strip()
        text = row.get("text", "").strip()
        corpus[row["_id"]] = (title + ". " + text) if title else text

    qrels = {}
    with open(os.path.join(folder, "qrels", split + ".tsv"), encoding="utf-8") as f:
        header = next(f)
        if not header.lower().startswith("query"):
            f.seek(0)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, grade = parts[0], parts[1], int(parts[2])
            if grade > 0:
                qrels.setdefault(qid, {})[did] = grade

    queries = {}
    for row in read_jsonl(os.path.join(folder, "queries.jsonl")):
        if row["_id"] in qrels:
            queries[row["_id"]] = row["text"]

    # A query with no judged relevant passage cannot be scored either way.
    qrels = {q: v for q, v in qrels.items() if q in queries}

    return corpus, queries, qrels
