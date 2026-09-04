"""Streamlit front end. Add article URLs a row at a time, index them, ask questions.

    streamlit run app.py

Nothing runs until you press a button. Articles pile up in a library so you can
keep adding sources and searching across all of them together.

Works with no API key. Without one the answers are built by pulling the best
matching sentences out of the retrieved passages.
"""

import streamlit as st

import chunk
import fetch
import generate
from retrieve import Dense, Hybrid, Reranker

st.set_page_config(page_title="MarketLens", layout="wide")
st.title("MarketLens")
st.caption("Ask questions across a set of news articles and see which passage "
           "each answer came from.")


# The sentence transformer takes about a minute to load the first time. Cache
# the model so flipping a setting does not pay that again. Only the model is
# cached, never the index, or adding an article would search the old one.
@st.cache_resource(show_spinner=False)
def load_dense_model(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


@st.cache_resource(show_spinner=False)
def load_cross_encoder(name):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)


state = st.session_state
state.setdefault("rows", [0])       # ids of the visible url boxes
state.setdefault("next_row", 1)
state.setdefault("library", [])     # articles fetched so far
state.setdefault("index", None)
state.setdefault("docs", [])
state.setdefault("sources", [])
state.setdefault("built_with", None)
state.setdefault("result", None)
state.setdefault("asked", "")


def add_row():
    state.rows.append(state.next_row)
    state.next_row += 1


def drop_row(row_id):
    key = "url_{}".format(row_id)
    if len(state.rows) > 1:
        state.rows.remove(row_id)
        state.pop(key, None)
    else:
        state[key] = ""


def drop_article(position):
    state.library.pop(position)
    state.built_with = None


def clear_library():
    state.library = []
    state.index = None
    state.docs = []
    state.sources = []
    state.built_with = None
    state.result = None


def show_text(value):
    # Streamlit reads $...$ as maths, which eats dollar amounts in a finance app.
    return value.replace("$", "\\$")


def build_index(documents, use_dense, use_rerank, size, overlap):
    docs, sources = chunk.chunk_documents(documents, size, overlap)

    dense = None
    if use_dense:
        dense = Dense()
        dense.model = load_dense_model(dense.model_name)

    reranker = None
    if use_rerank:
        reranker = Reranker()
        reranker.model = load_cross_encoder(reranker.model_name)

    index = Hybrid(use_bm25=True, use_dense=use_dense, rerank=use_rerank,
                   dense=dense, reranker=reranker).fit(docs)
    return index, docs, sources


with st.sidebar:
    st.header("Settings")
    use_dense = st.checkbox("Embedding search", value=True,
                            help="Downloads a small sentence transformer the first time.")
    use_rerank = st.checkbox("Cross encoder rerank", value=False,
                             help="More accurate, noticeably slower.")
    size = st.slider("Chunk size (words)", 100, 500, 200, step=50)
    overlap = st.slider("Chunk overlap (words)", 0, 200, 50, step=25)
    top_k = st.slider("Passages per answer", 1, 10, 5)

settings = (use_dense, use_rerank, size, overlap)

st.subheader("1. Add articles")

for row_id in list(state.rows):
    box, remove = st.columns([14, 1])
    box.text_input(
        "URL",
        key="url_{}".format(row_id),
        placeholder="https://www.example.com/markets/some-article",
        label_visibility="collapsed",
    )
    remove.button("X", key="drop_{}".format(row_id), on_click=drop_row,
                  args=(row_id,), help="Remove this row")

left, right = st.columns([1, 5])
left.button("Add another", on_click=add_row)
index_now = right.button("Fetch and index", type="primary")

if index_now:
    typed = []
    for row_id in state.rows:
        url = state.get("url_{}".format(row_id), "").strip()
        if url and url not in typed:
            typed.append(url)

    have = set(d["url"] for d in state.library)
    fresh = [u for u in typed if u not in have]

    if fresh:
        progress = st.progress(0.0)
        for i, url in enumerate(fresh):
            try:
                state.library.append(fetch.load_article(url))
            except Exception as err:
                st.error("{} -> {}".format(url, err))
            progress.progress((i + 1) / len(fresh))
        progress.empty()

    if not state.library:
        st.warning("Add at least one URL that loads.")
    else:
        with st.spinner("Building the index..."):
            index, docs, sources = build_index(state.library, *settings)
        state.index = index
        state.docs = docs
        state.sources = sources
        state.built_with = settings
        st.success("Indexed {} chunks from {} articles.".format(
            len(docs), len(state.library)))

if state.library:
    with st.expander("Library ({} articles)".format(len(state.library))):
        for position, doc in enumerate(state.library):
            name, remove = st.columns([14, 1])
            name.write("{}  \n{}".format(doc["title"] or "untitled", doc["url"]))
            remove.button("X", key="lib_drop_{}".format(position),
                          on_click=drop_article, args=(position,),
                          help="Remove this article")
        st.button("Clear library", on_click=clear_library)

if state.library and state.built_with != settings:
    st.info("The library or the settings changed since the last build. "
            "Press Fetch and index to rebuild.")

st.subheader("2. Ask")

# A form means the search only runs when the button is pressed. Without it
# Streamlit reruns the script on every widget change and the retrieval would
# fire again each time someone nudged a slider.
with st.form("ask"):
    question = st.text_input("Question",
                             placeholder="What did the company say about margins?")
    asked = st.form_submit_button("Search", type="primary")

if asked:
    if state.index is None:
        st.warning("Index some articles first.")
    elif not question.strip():
        st.warning("Type a question.")
    else:
        with st.spinner("Searching..."):
            state.result = generate.answer(
                question,
                state.index,
                state.docs,
                state.sources,
                k=top_k,
            )
        state.asked = question

if state.result:
    result = state.result
    st.markdown("#### Answer")
    st.caption("Question: {}".format(state.asked))
    st.write(show_text(result["answer"]))
    st.caption("Generated in {} mode using {}.".format(
        result["mode"], state.index.name()))

    st.markdown("#### Passages used")
    for item in result["sources"]:
        label = "{}. {} (score {})".format(
            item["rank"], item.get("title") or item.get("url", "chunk"),
            item["score"])
        with st.expander(label):
            st.write(show_text(item["text"]))
            if item.get("url"):
                st.caption(item["url"])
