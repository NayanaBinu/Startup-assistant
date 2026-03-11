import psycopg2
import numpy as np
from llama_index import SimpleDirectoryReader
from sentence_transformers import SentenceTransformer

# Load PDFs
documents = SimpleDirectoryReader("acts_pdfs").load_data()

# Embedding model (free)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="startup_assistant",
    user="postgres",
    password="yourpassword"
)
cur = conn.cursor()

for doc in documents:
    embedding = model.encode(doc.text).tolist()
    cur.execute(
        "INSERT INTO legal_docs (doc_id, section, content, embedding) VALUES (%s,%s,%s,%s)",
        (getattr(doc,'doc_id','Unknown'), getattr(doc,'section',''), doc.text, embedding)
    )

conn.commit()
cur.close()
conn.close()
print("✅ PDFs ingested with embeddings!")
