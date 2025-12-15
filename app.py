import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from rank_bm25 import BM25Okapi

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# --------------------------
# Paths
# --------------------------
INDEX_DIR = Path("index")
CHUNKS_STORE = INDEX_DIR / "chunks_store.pkl"

# --------------------------
# Text utilities
# --------------------------
def tokenize_ua(text: str) -> List[str]:
    return re.findall(r"[a-zа-яіїєґ0-9]+", text.lower())

def load_chunks() -> List[Dict[str, Any]]:
    with open(CHUNKS_STORE, "rb") as f:
        raw = pickle.load(f)

    chunks = []
    for ci in raw:
        text = getattr(ci, "text", None) or ci["text"]
        meta = getattr(ci, "metadata", None) or ci["metadata"]
        chunks.append({"text": text, "metadata": meta})
    return chunks

# --------------------------
# BM25
# --------------------------
def build_bm25(chunks):
    tokenized = [tokenize_ua(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)

def bm25_retrieve(bm25, chunks, query, k):
    qtok = tokenize_ua(query)
    scores = bm25.get_scores(qtok)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top if scores[i] > 0]

# --------------------------
# Dense
# --------------------------
def dense_retrieve(vs, query, k):
    docs = vs.similarity_search(query, k=k)
    return [{"text": d.page_content, "metadata": d.metadata} for d in docs]

# --------------------------
# Hybrid
# --------------------------
def hybrid_retrieve(bm25, vs, chunks, query, k):
    bm = bm25_retrieve(bm25, chunks, query, k)
    dn = dense_retrieve(vs, query, k)

    merged = {}
    for c in bm + dn:
        key = (c["metadata"].get("source"), c["metadata"].get("page"), c["text"][:80])
        merged[key] = c

    return list(merged.values())[:k]

# --------------------------
# Prompt
# --------------------------
def build_prompt(question, chunks):
    ctx = []
    for i, c in enumerate(chunks, start=1):
        m = c["metadata"]
        ctx.append(f"[{i}] {m.get('source','?')} {c['text']}")
    context = "\n\n".join(ctx)

    return f"""
Відповідай ТІЛЬКИ на основі контексту.
Став посилання [1], [2] у тексті відповіді.
Якщо відповіді немає — напиши: "Немає даних у документах."

Питання: {question}

Контекст:
{context}
""".strip()

# --------------------------
# UI
# --------------------------
def main():
    st.set_page_config(page_title="RAG QA (Ollama)", layout="wide")
    st.title("📚 RAG Question Answering (локально через Ollama)")
    st.info("UI завантажився ✅ Якщо щось піде не так — помилку буде показано тут")

    if not CHUNKS_STORE.exists():
        st.error("❌ Немає index/chunks_store.pkl → запусти python ingest.py")
        st.stop()

    chunks = load_chunks()

    # Sidebar
    st.sidebar.header("Налаштування")
    mode = st.sidebar.radio(
        "Retriever",
        ["Hybrid", "BM25", "Dense", "Без пошуку"],
        index=0
    )
    k = st.sidebar.slider("k (чанки)", 2, 10, 5)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)

    # Init components (SAFE)
    try:
        with st.spinner("Ініціалізація моделей…"):
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vectorstore = FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True
            )
            bm25 = build_bm25(chunks)
            llm = ChatOllama(model="mistral", temperature=temperature)
        st.success("Готово ✅")
    except Exception as e:
        st.error("❌ Помилка ініціалізації (Ollama не запущена?)")
        st.exception(e)
        st.stop()

    question = st.text_input(
        "Питання",
        placeholder="Напр.: Що робити, якщо відчувається запах гару?"
    )

    if st.button("🔎 Запитати") and question.strip():
        with st.spinner("Пошук і генерація відповіді…"):
            if mode == "BM25":
                retrieved = bm25_retrieve(bm25, chunks, question, k)
            elif mode == "Dense":
                retrieved = dense_retrieve(vectorstore, question, k)
            elif mode == "Hybrid":
                retrieved = hybrid_retrieve(bm25, vectorstore, chunks, question, k)
            else:
                retrieved = []

            prompt = build_prompt(question, retrieved) if retrieved else question
            answer = llm.invoke(prompt).content

        st.subheader("Відповідь")
        st.write(answer)

        if retrieved:
            st.subheader("Джерела")
            for i, c in enumerate(retrieved, start=1):
                m = c["metadata"]
                st.markdown(f"**[{i}] {m.get('source','?')}**")
                st.code(c["text"][:800], language="text")

# --------------------------
# ENTRY POINT
# --------------------------
if __name__ == "__main__":
    main()
