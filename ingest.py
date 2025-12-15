import re
import pickle
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

DATA_DIR = Path("data")
INDEX_DIR = Path("index")

BM25_STORE = INDEX_DIR / "bm25_store.pkl"
CHUNKS_STORE = INDEX_DIR / "chunks_store.pkl"

def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")

def normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", s)).strip()

def load_raw_docs(data_dir: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for p in data_dir.rglob("*"):
        if p.is_dir():
            continue
        ext = p.suffix.lower()

        if ext == ".pdf":
            pages = PyPDFLoader(str(p)).load()
            for d in pages:
                docs.append({
                    "text": d.page_content,
                    "metadata": {
                        "source": str(p).replace("\\", "/"),
                        "page": d.metadata.get("page", None),
                    }
                })
        elif ext in (".txt", ".md"):
            text = normalize_whitespace(read_text_file(p))
            docs.append({
                "text": text,
                "metadata": {
                    "source": str(p).replace("\\", "/"),
                    "page": None
                }
            })
    return docs

def main():
    if not DATA_DIR.exists():
        raise SystemExit("❌ Немає папки data/. Створи її та додай документи.")

    raw_docs = load_raw_docs(DATA_DIR)
    if not raw_docs:
        raise SystemExit("❌ Не знайдено документів у data/ (.pdf/.txt/.md)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    # робимо чанки як list[dict] (важливо!)
    chunks: List[Dict[str, Any]] = []
    for d in raw_docs:
        split_docs = splitter.create_documents([d["text"]], metadatas=[d["metadata"]])
        for sd in split_docs:
            chunks.append({
                "text": sd.page_content,
                "metadata": dict(sd.metadata)
            })

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Dense (FAISS)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    lc_docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]
    vectorstore = FAISS.from_documents(lc_docs, embeddings)
    vectorstore.save_local(str(INDEX_DIR))

    # Store chunks (dicts, NOT ChunkItem objects)
    with open(CHUNKS_STORE, "wb") as f:
        pickle.dump(chunks, f)

    with open(BM25_STORE, "wb") as f:
        pickle.dump({"ok": True, "count": len(chunks)}, f)

    print("✅ Готово.")
    print(f"   Документів: {len(raw_docs)}")
    print(f"   Чанків:     {len(chunks)}")
    print(f"   Індекс:     {INDEX_DIR}/ (FAISS + BM25 store)")

if __name__ == "__main__":
    main()
