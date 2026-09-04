"""Splitting documents into retrievable chunks.

Chunk size is the setting that quietly decides how good a RAG system is.
Too small and a chunk holds a fact without the context that makes it
findable. Too large and the embedding averages several topics together and
matches nothing well. The overlap exists so an answer that straddles a
boundary still lands whole inside at least one chunk.

evaluate.py sweeps these numbers rather than guessing them.
"""

import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    parts = [s.strip() for s in SENTENCE_END.split(text) if s.strip()]
    return parts or [text.strip()]


def chunk_text(text, size=200, overlap=50):
    """Splits into chunks of about `size` words, overlapping by `overlap`.

    Splits on sentence boundaries so a chunk never ends mid sentence, then
    packs sentences until the word budget runs out.
    """
    if overlap >= size:
        raise ValueError("overlap has to be smaller than size")

    sentences = split_sentences(text)
    chunks = []
    current = []
    count = 0

    for sentence in sentences:
        words = len(sentence.split())

        if count + words > size and current:
            chunks.append(" ".join(current))
            # Walk back from the end until the overlap budget is used up.
            kept = []
            kept_words = 0
            for previous in reversed(current):
                previous_words = len(previous.split())
                if kept_words + previous_words > overlap:
                    break
                kept.insert(0, previous)
                kept_words += previous_words
            current = kept
            count = kept_words

        current.append(sentence)
        count += words

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_documents(documents, size=200, overlap=50):
    """documents is a list of dicts with title and text.

    Returns the chunk strings and a matching list of where each came from, so
    an answer can cite its source.
    """
    texts = []
    sources = []

    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"], size, overlap)):
            # The title goes in front of every chunk. Later chunks in an
            # article often use "the company" instead of the name, and without
            # the title they are unfindable by keyword search.
            title = doc.get("title", "")
            texts.append((title + ". " + chunk) if title else chunk)
            sources.append({
                "url": doc.get("url", ""),
                "title": title,
                "chunk": i,
            })

    return texts, sources
