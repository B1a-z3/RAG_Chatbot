"""
ingest.py
Loads docs from ./data (PDF or .md), chunks them, embeds locally, stores in ChromaDB.

Run: python src/ingest.py
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "product_docs"

CHUNK_SIZE = 500       # tokens (approx, using word count as proxy)
CHUNK_OVERLAP = 50


def load_documents():
    """Load all PDF and Markdown files from the data directory."""
    docs = []

    for path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        docs.append({"source": os.path.basename(path), "text": text})

    for path in glob.glob(os.path.join(DATA_DIR, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})

    if not docs:
        raise FileNotFoundError(
            f"No .pdf or .md files found in {DATA_DIR}. Add your product docs there first."
        )
    return docs


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple word-based sliding window chunker. Swap for a token-aware
    splitter (e.g. tiktoken) if you need exact token counts."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index():
    docs = load_documents()

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # fast, local, 384-dim, no API cost
    )

    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh build each run — drop existing collection if present
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['source']}_{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_index": i})

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Indexed {len(texts)} chunks from {len(docs)} documents into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()
