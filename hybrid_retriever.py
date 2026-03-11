import re
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# =============================
# CONFIG
# =============================

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DENSE_TOP_K = 400
FINAL_TOP_K = 100
BM25_TOP_K = 400
DENSE_WEIGHT = 0.55
BM25_WEIGHT = 0.30
SECTION_BOOST = 0.10
ACT_BOOST = 0.05

# =============================
# INIT
# =============================

embedder = SentenceTransformer(EMBED_MODEL)

conn = psycopg2.connect(
    host="localhost",
    database="startup_assistant",
    user="postgres",
    password="300234",
    port="5432"
)
cur = conn.cursor()

# Load full corpus for BM25
cur.execute("SELECT chunk_id, act_name, section, content FROM legal_docs;")
rows = cur.fetchall()

chunk_ids = [r[0] for r in rows]
acts = {r[0]: r[1] for r in rows}
sections = {r[0]: r[2] for r in rows}
corpus = [r[3].lower() for r in rows]

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

print("Production Legal Retriever Ready.")
def extract_section_number(query):
    match = re.search(r'(\d+[A-Za-z]*)', query)
    return match.group(1) if match else None

def hybrid_retrieve(query, top_k=50):

    query_lower = query.lower()
    section_number = extract_section_number(query)

    # ==============================
    # 1️⃣ Dense Retrieval
    # ==============================
    query_embedding = embedder.encode(
        "Represent this sentence for searching relevant passages: " + query,
        normalize_embeddings=True
    ).tolist()

    cur.execute("""
    SELECT chunk_id, act_name, section, content,
           1 - (embedding <=> %s::vector) AS dense_score
    FROM legal_docs
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
""", (query_embedding, query_embedding, DENSE_TOP_K))


    dense_results = cur.fetchall()

    dense_dict = {
        r[0]: {
            "act": r[1],
            "section": r[2],
            "content": r[3],
            "dense_score": r[4]
        }
        for r in dense_results
    }

    # ==============================
    # 2️⃣ BM25 Retrieval
    # ==============================
    tokenized_query = query_lower.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Take top 150 BM25 separately
    bm25_top_indices = np.argsort(bm25_scores)[-BM25_TOP_K:][::-1]


    bm25_dict = {}
    for idx in bm25_top_indices:
        chunk_id = chunk_ids[idx]
        bm25_dict[chunk_id] = {
            "act": acts[chunk_id],
            "section": sections[chunk_id],
            "content": corpus[idx],
            "bm25_score": bm25_scores[idx]
        }

    # ==============================
    # 3️⃣ Candidate Union
    # ==============================
    candidate_ids = set(dense_dict.keys()).union(set(bm25_dict.keys()))

    final_results = []

    for cid in candidate_ids:

        dense_score = dense_dict.get(cid, {}).get("dense_score", 0)
        bm_score = bm25_dict.get(cid, {}).get("bm25_score", 0)

        boost = 0

        if section_number and section_number in sections[cid].lower():
            boost += 0.15

        if acts[cid].lower() in query_lower:
            boost += 0.05

        final_score = (0.6 * dense_score) + (0.4 * bm_score) + boost

        final_results.append({
            "chunk_id": cid,
            "act": acts[cid],
            "section": sections[cid],
            "content": dense_dict.get(cid, bm25_dict.get(cid))["content"],
            "final_score": final_score
        })

    final_results.sort(key=lambda x: x["final_score"], reverse=True)

    return final_results[:FINAL_TOP_K]

