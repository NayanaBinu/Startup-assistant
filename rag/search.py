import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# 1. Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2. Connect to Postgres
conn = psycopg2.connect(
    dbname="startup_assistant",
    user="postgres",
    password="300234",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

def search_and_rerank(query, top_k=3):
    # Embed query
    q_emb = embedder.encode(query).tolist()

    # 3. Get candidate docs (limit more than top_k, e.g., 20)
    cur.execute("SELECT id, section, content, embedding FROM legal_docs LIMIT 20")
    rows = cur.fetchall()

    # 4. Rerank with cosine similarity
    scored = []
    for r in rows:
        emb = np.array(r[3], dtype=float)
        score = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        scored.append((score, r[1], r[2]))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]

    return top

def ask_ollama(query, top_k=3):
    top_contexts = search_and_rerank(query, top_k)
    context_text = "\n\n".join([f"[{s}] {c}" for _, s, c in top_contexts])

    prompt = f"""
    You are a legal assistant. Answer the following query using the provided context only.

    Query: {query}

    Context:
    {context_text}

    Final Answer:
    """

    response = ollama.chat(model="mistral", messages=[
        {"role": "user", "content": prompt}
    ])
    print(response["message"]["content"])

# Example query
if __name__ == "__main__":
    ask_ollama("What are the compliance rules for startups in India?", top_k=3)
