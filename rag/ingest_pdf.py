import os
import re
import psycopg2
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ===============================
# CONFIG
# ===============================

PDF_FOLDER = "acts_pdfs"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

DB_CONFIG = {
    "dbname": "startup_assistant",
    "user": "postgres",
    "password": "300234",
    "host": "localhost",
    "port": 5432
}

# ===============================
# LOAD MODEL
# ===============================

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# ===============================
# CLEANING FUNCTIONS
# ===============================

def clean_legal_text(text):
    """
    Remove amendment notes, footnotes, OCR garbage.
    """

    # Remove amendment lines
    text = re.sub(r"Ins\. by Act.*?\)", "", text)
    text = re.sub(r"Subs\. by Act.*?\)", "", text)
    text = re.sub(r"w\.e\.f\..*?\)", "", text)

    # Remove footnote markers like 1[ or 2[
    text = re.sub(r"\d+\[", "", text)

    # Fix broken words (OCR split)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_sections(text):
    """
    Extract full sections cleanly.
    """
    pattern = re.compile(r"\n\s*(\d+[A-Z]?\.\s+[^\n]{3,200})")
    matches = list(pattern.finditer(text))

    sections = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        title = match.group(1).strip()
        content = text[start:end].strip()

        content = clean_legal_text(content)

        if len(content) > 200:
            sections.append((title, content))

    return sections


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ===============================
# INGESTION
# ===============================

def ingest_pdf(pdf_path):

    act_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"Processing {act_name}...")

    raw_text = extract_text_from_pdf(pdf_path)
    sections = extract_sections(raw_text)

    inserted = 0

    for section_title, section_text in sections:

        embedding = embedder.encode(
            section_text,
            normalize_embeddings=True
        ).tolist()

        cur.execute("""
            INSERT INTO legal_docs
            (act_name, section, content, embedding)
            VALUES (%s, %s, %s, %s)
        """, (act_name, section_title, section_text, embedding))

        inserted += 1

    conn.commit()
    print(f"Inserted {inserted} clean sections.")


# ===============================
# RUN
# ===============================

if __name__ == "__main__":

    print("Clearing old data...")
    cur.execute("DELETE FROM legal_docs;")
    conn.commit()

    for file in os.listdir(PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            ingest_pdf(os.path.join(PDF_FOLDER, file))

    cur.close()
    conn.close()

    print("Production ingestion complete.")
