import os
import psycopg2
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, CrossEncoder
from mistralai import Mistral

# ---------------- CONFIG ---------------- #

DB_CONFIG = {
    "dbname": "startup_assistant",
    "user": "postgres",
    "password": "300234",
    "host": "localhost",
    "port": 5432
}

EMBED_MODEL = "BAAI/bge-large-en"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE = 50
TOP_K_FINAL = 5

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# ---------------- INIT ---------------- #

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading reranker...")
reranker = CrossEncoder(RERANK_MODEL)

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

if not MISTRAL_API_KEY:
    raise Exception("Set MISTRAL_API_KEY environment variable")

mistral = Mistral(api_key=MISTRAL_API_KEY)

app = Flask(__name__)

# ---------------- RETRIEVAL ---------------- #

def retrieve(query):
    query_embedding = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    cur.execute("""
        SELECT doc_id, act_name, section, content,
               1 - (embedding <=> %s) AS similarity
        FROM legal_docs
        ORDER BY embedding <=> %s
        LIMIT %s;
    """, (query_embedding, query_embedding, TOP_K_RETRIEVE))

    return cur.fetchall()


def rerank(query, docs):
    pairs = [(query, doc[3]) for doc in docs]
    scores = reranker.predict(pairs)

    scored = list(zip(scores, docs))
    scored.sort(reverse=True, key=lambda x: x[0])

    return [doc for score, doc in scored[:TOP_K_FINAL]]


def build_context(docs):
    context = ""
    for doc in docs:
        context += f"""
Act: {doc[1]}
Section: {doc[2]}

{doc[3]}

"""
    return context


def call_llm(prompt):
    response = mistral.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def rag_answer(query):
    retrieved = retrieve(query)
    top_docs = rerank(query, retrieved)
    context = build_context(top_docs)

    prompt = f"""
You are a legal expert specializing in Indian startup law.

Rules:
- Use ONLY the provided context.
- Do not assume or invent.
- Cite section numbers clearly.
- If answer not found, respond exactly:
"The provided documents do not contain sufficient information."

Question:
{query}

Context:
{context}

Answer:
"""

    answer = call_llm(prompt)

    sources = [
        {
            "act": doc[1],
            "section": doc[2]
        }
        for doc in top_docs
    ]

    return answer, sources

# ---------------- API ---------------- #

@app.route("/api/rag/qa", methods=["POST"])
def rag_qa():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query required"}), 400

    answer, sources = rag_answer(query)

    return jsonify({
        "query": query,
        "answer": answer,
        "sources": sources
    })

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)