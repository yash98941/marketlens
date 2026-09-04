"""Turning retrieved passages into an answer.

Two modes on purpose:

  extractive  picks the sentences from the retrieved passages that best match
              the question and stitches them together. No API key, no model
              download, runs anywhere. This is the default so the project can
              be cloned and run without signing up for anything.

  llm         sends the passages to a chat model with instructions to answer
              only from them. Used if OPENAI_API_KEY is set.

The extractive mode is also useful as a control. If the answers look fine
without a language model then the retrieval is doing the work, and if they do
not then no amount of prompt tuning will save it.
"""

import os
import re

from retrieve import tokenise

SYSTEM = (
    "Answer the question using only the passages given. "
    "Quote numbers exactly as they appear. "
    "If the passages do not contain the answer, say so instead of guessing."
)


def split_sentences(text):
    # Collapse the newlines first, or a chunk that spans a paragraph break
    # comes back as one ragged sentence with a gap in the middle of it.
    text = re.sub(r"\s+", " ", text)
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return parts or [text.strip()]


def extractive_answer(question, passages, max_sentences=3):
    """Scores every sentence by overlap with the question and keeps the best."""
    wanted = set(tokenise(question))
    if not wanted:
        return "Ask something with a bit more in it."

    scored = []
    for rank, text in enumerate(passages):
        for sentence in split_sentences(text):
            words = set(tokenise(sentence))
            if not words:
                continue
            overlap = len(wanted & words) / len(wanted)
            # Nudge sentences from better ranked passages ahead of equal
            # matches further down the list.
            scored.append((overlap - 0.01 * rank, sentence))

    scored.sort(reverse=True)
    picked = [s for score, s in scored[:max_sentences] if score > 0]

    if not picked:
        return "Nothing in the retrieved passages answers that."
    return " ".join(picked)


def llm_answer(question, passages, model="gpt-4o-mini"):
    from openai import OpenAI

    context = "\n\n".join(
        "[{}] {}".format(i + 1, text) for i, text in enumerate(passages))

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": "Passages:\n{}\n\nQuestion: {}".format(context, question)},
        ],
    )
    return response.choices[0].message.content.strip()


def answer(question, retriever, docs, sources=None, k=5, mode="auto"):
    """Retrieves then answers, and hands back what it used.

    Returning the passages is not decoration. An answer with no way to check
    it against the source is worth very little, and citations are the cheapest
    guard against the model making something up.
    """
    hits = retriever.search(question, k=k)
    passages = [docs[i] for i, _ in hits]

    if mode == "auto":
        mode = "llm" if os.environ.get("OPENAI_API_KEY") else "extractive"

    if mode == "llm":
        text = llm_answer(question, passages)
    else:
        text = extractive_answer(question, passages)

    used = []
    for rank, (doc_id, score) in enumerate(hits):
        item = {"rank": rank + 1, "score": round(score, 4),
                "text": docs[doc_id][:400]}
        if sources:
            item.update(sources[doc_id])
        used.append(item)

    return {"answer": text, "mode": mode, "sources": used}
