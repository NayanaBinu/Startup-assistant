import requests
import json
import time
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# ===============================
# CONFIGURATION
# ===============================

API_URL = "http://localhost:5000/api/rag/qa"
TOP_K = 5
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

print("Loading evaluation embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

# ===============================
# METRIC FUNCTIONS
# ===============================

def hit_rate(sources, expected_sections):
    for s in sources[:TOP_K]:
        if any(sec in str(s["section"]) for sec in expected_sections):
            return 1
    return 0


def reciprocal_rank(sources, expected_sections):
    for i, s in enumerate(sources[:TOP_K]):
        if any(sec in str(s["section"]) for sec in expected_sections):
            return 1 / (i + 1)
    return 0


def citation_accuracy(answer, expected_sections):
    found_sections = re.findall(r"Section\s+([\dA-Za-z()]+)", answer)

    for sec in expected_sections:
        if any(sec in found for found in found_sections):
            return 1
    return 0


def semantic_relevance(answer, reference_answer):
    if not reference_answer:
        return 0

    emb1 = embedder.encode(answer, convert_to_tensor=True)
    emb2 = embedder.encode(reference_answer, convert_to_tensor=True)

    return float(util.cos_sim(emb1, emb2))


def answer_groundedness(answer, context_sources):
    context_text = " ".join([
        s.get("content", "") for s in context_sources
    ])

    answer_tokens = set(answer.lower().split())
    context_tokens = set(context_text.lower().split())

    if len(answer_tokens) == 0:
        return 0

    overlap = len(answer_tokens.intersection(context_tokens))
    return overlap / len(answer_tokens)


# ===============================
# MAIN EVALUATION
# ===============================

def evaluate_rag():
    with open("legal_rag_test_set.json", "r") as f:
        test_data = json.load(f)

    all_results = []

    for idx, item in enumerate(test_data):
        question = item["question"]
        expected_sections = item["expected_section"]
        reference_answer = item.get("reference_answer", "")

        print(f"Evaluating Q{idx+1}: {question}")

        start_time = time.time()
        response = requests.post(API_URL, json={"query": question})
        latency = time.time() - start_time

        if response.status_code != 200:
            print("API error")
            continue

        data = response.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])

        # ---------------- Metrics ----------------

        hr = hit_rate(sources, expected_sections)
        rr = reciprocal_rank(sources, expected_sections)
        citation_acc = citation_accuracy(answer, expected_sections)
        sem_rel = semantic_relevance(answer, reference_answer)
        groundedness = answer_groundedness(answer, sources)

        all_results.append({
            "question": question,
            "hit_rate": hr,
            "mrr": rr,
            "citation_accuracy": citation_acc,
            "semantic_relevance": sem_rel,
            "groundedness": groundedness,
            "latency": latency
        })

    # ===============================
    # AGGREGATED METRICS
    # ===============================

    df = pd.DataFrame(all_results)

    print("\n==============================")
    print("📊 REALISTIC RAG EVALUATION REPORT")
    print("==============================")

    print(f"Total Questions: {len(df)}")
    print(f"Hit Rate@{TOP_K}: {df['hit_rate'].mean():.2f}")
    print(f"MRR: {df['mrr'].mean():.2f}")
    print(f"Citation Accuracy: {df['citation_accuracy'].mean():.2f}")
    print(f"Semantic Relevance: {df['semantic_relevance'].mean():.2f}")
    print(f"Groundedness: {df['groundedness'].mean():.2f}")
    print(f"Average Latency: {df['latency'].mean():.2f} sec")

    df.to_csv("rag_evaluation_results_v3.csv", index=False)
    print("\nResults saved to rag_evaluation_results_v3.csv")


if __name__ == "__main__":
    evaluate_rag()
