"""
evaluate.py
Measures retrieval quality and answer faithfulness against eval/qa_testset.json.

Run: python eval/evaluate.py
"""

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import retrieve, answer_question  # noqa: E402

TESTSET_PATH = os.path.join(os.path.dirname(__file__), "qa_testset.json")


def load_testset():
    with open(TESTSET_PATH, "r") as f:
        return json.load(f)


def evaluate_retrieval(testset, top_k=3):
    """Precision@k: did the expected source appear in the top-k retrieved chunks?"""
    hits = 0
    for item in testset:
        chunks = retrieve(item["question"], top_k=top_k)
        retrieved_sources = {c["source"] for c in chunks}
        if item["expected_source"] in retrieved_sources:
            hits += 1
        else:
            print(f"[MISS] '{item['question']}' → expected {item['expected_source']}, "
                  f"got {retrieved_sources}")
    precision = hits / len(testset)
    print(f"\nRetrieval precision@{top_k}: {precision:.2%} ({hits}/{len(testset)})")
    return precision


def evaluate_answers(testset):
    """Simple keyword-containment check as a proxy for answer faithfulness.
    For a rigorous eval, swap this for RAGAS or an LLM-as-judge scorer."""
    hits = 0
    total_cost = 0
    total_latency = 0
    for item in testset:
        result = answer_question(item["question"])
        total_cost += result["cost_usd"]
        total_latency += result["latency_seconds"]
        if item["expected_answer_contains"].lower() in result["answer"].lower():
            hits += 1
        else:
            print(f"[WEAK ANSWER] '{item['question']}' → {result['answer'][:120]}...")

    accuracy = hits / len(testset)
    print(f"\nAnswer keyword accuracy: {accuracy:.2%} ({hits}/{len(testset)})")
    print(f"Avg cost per query: ${total_cost/len(testset):.5f}")
    print(f"Avg latency per query: {total_latency/len(testset):.2f}s")


if __name__ == "__main__":
    testset = load_testset()
    print("=== Retrieval Evaluation ===")
    evaluate_retrieval(testset)
    print("\n=== Answer Evaluation ===")
    evaluate_answers(testset)
