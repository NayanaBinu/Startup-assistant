from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import json

# Load legal data
with open("companies_act_sections_filled_retry.json", "r") as f:
    sections = json.load(f)

# Embed model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB setup
client = PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("companies_act")

# Optional: clear existing
ids = collection.get()["ids"]
if ids:
    collection.delete(ids=ids)

# Add data
for i, sec in enumerate(sections):
    doc_id = f"section-{i}"
    content = f"{sec['section']} - {sec['title']}\n{sec['text']}"
    collection.add(
        documents=[content],
        ids=[doc_id],
        metadatas=[{"section": sec["section"], "title": sec["title"]}]
    )

client.persist()
print("✅ Legal data embedded and stored successfully.")
