"""
rag.py
Core retrieval + generation pipeline. Import `answer_question()` from the UI or eval script.
"""

import os
import time
import chromadb
from chromadb.utils import embedding_functions
import ollama

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "product_docs"
MODEL = "llama3.1"
TOP_K = 4

SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's question using ONLY \
the provided context excerpts from the product documentation.

Rules:
- If the context does not contain enough information to answer, say so explicitly — \
do not guess or use outside knowledge.
- Cite which source document each part of your answer comes from.
- Keep answers concise and direct.
"""

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _get_collection():
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    return chroma_client.get_collection(COLLECTION_NAME, embedding_function=_embedding_fn)


def retrieve(query, top_k=TOP_K):
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"text": doc, "source": meta["source"], "score": 1 - dist})
    return chunks


def build_prompt(query, chunks):
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"Context:\n{context}\n\nQuestion: {query}"


def answer_question(query, top_k=TOP_K):
    start = time.time()

    chunks = retrieve(query, top_k=top_k)
    user_prompt = build_prompt(query, chunks)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    latency = time.time() - start
    input_tokens = response.get("prompt_eval_count", 0)
    output_tokens = response.get("eval_count", 0)

    return {
        "answer": response["message"]["content"],
        "sources": [{"source": c["source"], "score": round(c["score"], 3)} for c in chunks],
        "latency_seconds": round(latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": 0.0,
    }


if __name__ == "__main__":
    result = answer_question("What is this product's refund policy?")
    print(result["answer"])
    print(result["sources"])
