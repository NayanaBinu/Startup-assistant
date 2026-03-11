from flask import Flask, request, jsonify
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
conn = psycopg2.connect(
    host="localhost",
    database="startup_assistant",
    user="postgres",
    password="300234"
)
cur = conn.cursor()

def cosine(a,b):
    a,b = np.array(a), np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json.get("query")
    query_vec = model.encode(query)
    cur.execute("SELECT doc_id, section, content, embedding FROM legal_docs")
    rows = cur.fetchall()
    top = sorted([(cosine(query_vec,row[3]), row) for row in rows], key=lambda x:x[0], reverse=True)[:5]
    results = [{"doc_id":r[1][0], "section":r[1][1], "content":r[1][2][:500]} for r in top]
    return jsonify({"answer":"Top matching sections","results":results})

if __name__=="__main__":
    app.run(debug=True, port=5000)
