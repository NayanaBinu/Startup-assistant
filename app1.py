import os
import re
import logging
import tempfile
import numpy as np
from mistralai import Mistral
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")


from flask import Flask, request, jsonify, session, render_template, send_from_directory,send_file
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import IntegrityError
from DocsGenerator.generator import (
    generate_nda,
    generate_pitch_deck,
    generate_mou,
    generate_rti
)


from models import db, User, Startup
from matcher import load_schemes, match_schemes

# --- NEW: imports for document summarizer ---
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import ollama

# --- NEW: imports for RAG (semantic search over legal_docs) ---
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- OPTIONAL: Mistral API (Cloud LLM) ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

mistral_client = None
if MISTRAL_API_KEY:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    logger.info("✅ Mistral API enabled (cloud)")
else:
    logger.info("⚠️ MISTRAL_API_KEY not set → using Ollama (local)")


# --- App Initialization ---
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, supports_credentials=True)   # allow cookies/sessions

# --- Configuration ---
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "a-strong-default-secret-key-for-dev")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:300234@localhost:5432/startup_assistant"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# --- NEW: Tesseract path (adjust if installed elsewhere) ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- NEW: helper to extract text from PDFs / images ---
def extract_text_from_file(file_path, file_ext):
    text = ""
    try:
        if file_ext.lower() == ".pdf":
            # Convert PDF pages to images
            pages = convert_from_path(
                file_path,
                poppler_path=r"C:\poppler-windows-25.07.0-0\poppler-25.07.0\Library\bin"  # update if needed
            )
            for page in pages:
                text += pytesseract.image_to_string(page)
        else:
            # Process as image
            text = pytesseract.image_to_string(Image.open(file_path))
    except Exception as e:
        logger.exception(f"Error extracting text: {e}")
        text = ""
    return text.strip()


# --- NEW: RAG initialization (semantic search over legal_docs) ---
# SentenceTransformer model for embeddings
rag_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Direct psycopg2 connection for the legal_docs table
rag_conn = psycopg2.connect(
    host="localhost",
    database="startup_assistant",
    user="postgres",
    password="300234",
    port="5432",
)
rag_cur = rag_conn.cursor()


def cosine_similarity(a, b):
    """Safely compute cosine similarity between two vectors."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rag_search(query, top_k=5):
    q_emb = rag_model.encode(query)

    # Extract section number
    sec_match = re.search(r'section\s+(\d+[A-Z]?)', query, re.IGNORECASE)
    section_hint = sec_match.group(1) if sec_match else None

    # Detect Act from query
    act = None
    if "companies act" in query.lower():
        act = "CompaniesAct"
    elif "gst" in query.lower():
        act = "GST"
    elif "llp" in query.lower():
        act = "Limited liability"

    # Build SQL query
    sql = "SELECT id, doc_id, section, content, embedding FROM legal_docs WHERE 1=1"
    params = []

    if act:
        sql += " AND doc_id ILIKE %s"
        params.append(f"%{act}%")

    if section_hint:
        sql += " AND section LIKE %s"
        params.append(f"{section_hint}.%")

    rag_cur.execute(sql, params)
    rows = rag_cur.fetchall()

    if not rows:
        return []

    # Semantic scoring
    scored = []
    for r in rows:
        emb = np.array(r[4], dtype=float)
        score = cosine_similarity(q_emb, emb)
        scored.append((score, r))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]

    return [
        {
            "score": score,
            "doc_id": row[1],
            "section": row[2],
            "content": row[3][:600]
        }
        for score, row in top
    ]

def call_llm(messages, model="mistral-small-latest"):
    """
    Unified LLM caller:
    - Uses Mistral API if available
    - Falls back to Ollama if not
    """

    # ✅ Cloud LLM
    if mistral_client:
        response = mistral_client.chat.complete(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content.strip()

    # ✅ Local fallback (existing behavior)
    response = ollama.chat(
        model="mistral",
        messages=messages
    )
    return response["message"]["content"].strip()

def rag_answer_with_llm(query, top_k=3):
    """
    RAG pipeline:
    - retrieve top_k relevant legal sections
    - generate final answer using Ollama (Mistral)
    """

    retrieved = rag_search(query, top_k=top_k)

    if not retrieved:
        return "I could not find relevant legal information in the database."

    context_text = "\n\n".join(
        [f"Section: {r['section']}\n{r['content']}" for r in retrieved]
    )

    prompt = f"""
You are a legal assistant for startups in India.

Answer the user's question using ONLY the context below.
If the answer is not present in the context, say:
"I could not find this information in the provided legal documents."

Question:
{query}

Context:
{context_text}

Answer:
"""

    return call_llm(
    messages=[{"role": "user", "content": prompt}],
    model="mistral-small-latest"
)



# Load schemes once at startup (ensure path correct)
SCHEMES_FILE = os.path.join(os.path.dirname(__file__), "startup_schemes_final.json")
try:
    schemes = load_schemes(SCHEMES_FILE)
    logger.info(f"Loaded {len(schemes)} schemes from {SCHEMES_FILE}")
except Exception as e:
    schemes = []
    logger.exception(f"Failed to load schemes from {SCHEMES_FILE}: {e}")

# --------------------------
# Helper / Debug route to list registered routes
# --------------------------
@app.route("/_routes")
def show_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "rule": str(rule),
            "methods": sorted(list(rule.methods))
        })
    return jsonify({"routes": routes})

# --- API Routes ---
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    required_fields = ["full_name", "email", "password", "startup_name", "domain", "registration_type", "stage"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        if User.query.filter_by(email=data["email"]).first():
            return jsonify({"error": "Email already registered"}), 409

        new_user = User(
            full_name=data["full_name"],
            email=data["email"].lower(),
            password_hash=generate_password_hash(data["password"])
        )
        db.session.add(new_user)
        db.session.flush()

        new_startup = Startup(
            user_id=new_user.user_id,
            startup_name=data["startup_name"],
            domain=data["domain"],
            registration_type=data["registration_type"],
            stage=data["stage"],
            funding_amount=data.get("funding_amount"),
            team_size=data.get("team_size"),
            location=data.get("location"),
            website=data.get("website"),
            problem_statement=data.get("problem_statement"),
            vision=data.get("vision")
        )
        db.session.add(new_startup)
        db.session.commit()

        return jsonify({"message": "Signup successful"}), 201

    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "A database integrity error occurred."}), 500
    except Exception as e:
        db.session.rollback()
        logger.exception("An unexpected error occurred during signup")
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password are required"}), 400

    user = User.query.filter_by(email=data["email"].lower()).first()

    if user and check_password_hash(user.password_hash, data["password"]):
        session["user_id"] = user.user_id
        return jsonify({
            "message": "Login successful",
            "user_id": user.user_id
        }), 200

    return jsonify({"error": "Invalid email or password"}), 401


# Serve main index (templates/index.html expected)
@app.route("/")
def index():
    return render_template("index.html")


# route for login.html page (GET)
@app.route("/login.html")
def login_html():
    return render_template("login.html")


# Scheme matcher page (templates/scheme_matcher.html expected)
@app.route("/scheme-matcher")
def scheme_matcher_page():
    return render_template("scheme_matcher.html")


# route for scheme_matcher.html (if you link directly to it)
@app.route("/scheme_matcher.html")
def scheme_matcher_html():
    return render_template("scheme_matcher.html")


# Provide both endpoints in case frontend expects /match or /api/match
@app.route("/match", methods=["POST"])
@app.route("/api/match", methods=["POST"])
def match_route():
    try:
        data = request.get_json() or {}
        domain = data.get("domain")
        registration = data.get("registration") or data.get("registration_type")
        stage = data.get("stage")
        results = match_schemes(schemes, domain, registration, stage)
        return jsonify(results)
    except Exception as e:
        logger.exception("Error while processing match request")
        return jsonify({"error": "Server error while matching schemes"}), 500


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logout successful"}), 200


@app.route("/user/<int:user_id>", methods=["GET"])
def get_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    startup = Startup.query.filter_by(user_id=user_id).first()

    return jsonify({
        "user_id": user.user_id,
        "full_name": user.full_name,
        "email": user.email,
        "startup": {
            "startup_name": startup.startup_name if startup else None,
            "domain": startup.domain if startup else None,
            "registration_type": startup.registration_type if startup else None,
            "stage": startup.stage if startup else None,
            "funding_amount": startup.funding_amount if startup else None,
            "team_size": startup.team_size if startup else None,
            "location": startup.location if startup else None,
            "website": startup.website if startup else None,
            "problem_statement": startup.problem_statement if startup else None,
            "vision": startup.vision if startup else None
        }
    }), 200


@app.route("/current_user", methods=["GET"])
def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"logged_in": False}), 200

    user = User.query.get(user_id)
    if not user:
        return jsonify({"logged_in": False}), 200

    return jsonify({
        "logged_in": True,
        "user_id": user.user_id,
        "full_name": user.full_name,
        "email": user.email
    }), 200


@app.route("/dashboard.html")
def dashboard_html():
    return render_template("dashboard.html")


@app.route("/index.html")
def index_html_alias():
    # Serve the same index page for /index.html
    return render_template("index.html")


@app.route("/styles.css")
def styles_css():
    # Serve styles.css from the /static folder
    return app.send_static_file("styles.css")


@app.route("/api/match/auto", methods=["GET"])
def match_for_current_user():
    """Match schemes using the startup details of the logged-in user."""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    startup = Startup.query.filter_by(user_id=user_id).first()
    if not startup:
        return jsonify({"error": "Startup profile not found for this user"}), 404

    domain = startup.domain
    registration = startup.registration_type
    stage = startup.stage

    results = match_schemes(
        schemes,
        domain=domain,
        registration=registration,
        stage=stage
    )

    return jsonify({
        "criteria": {
            "domain": domain,
            "registration": registration,
            "stage": stage,
        },
        "results": results
    })


# --- Doc summarizer page route ---
@app.route("/doc-summarizer")
def doc_summarizer_page():
    # expects templates/doc_summarizer.html
    return render_template("doc_summarizer.html")


# support /doc_summarizer.html directly
@app.route("/doc_summarizer.html")
def doc_summarizer_html_alias():
    return render_template("doc_summarizer.html")


# --- summarization endpoint ---
@app.route("/summarize", methods=["POST"])
def summarize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'})

    # Save temporarily
    suffix = os.path.splitext(file.filename)[1]  # preserve file extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        file_path = tmp.name

    # Extract text using OCR / PDF conversion
    file_ext = os.path.splitext(file.filename)[1]
    text = extract_text_from_file(file_path, file_ext)

    if not text:
        os.remove(file_path)
        return jsonify({'error': 'No text detected. Try a clearer scan or a text-based PDF.'})

    # Summarize using Mistral via Ollama
    try:
        summary = call_llm(
    messages=[
        {
            "role": "system",
            "content": "You are a legal assistant. Summarize the given legal text clearly and concisely."
        },
        {"role": "user", "content": text}
    ],
    model="mistral-small-latest"
)

    except Exception as e:
        logger.exception("Error contacting Ollama for summarization")
        summary = f"❌ Error contacting Ollama: {e}"

    # Clean up temp file
    os.remove(file_path)
    return jsonify({'summary': summary})


# --- NEW: RAG API endpoint (no UI) ---
# --- RAG API endpoint (no UI) ---
@app.route("/ask", methods=["POST"])
@app.route("/api/rag/ask", methods=["POST"])
def ask_rag():
    """
    Simple RAG endpoint:
    - Input: JSON { "query": "..." }
    - Output: top matching sections from legal_docs
    """
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    results = rag_search(query, top_k=5)

    return jsonify({
        "answer": "Top matching sections",
        "results": [
            {
                "doc_id": r["doc_id"],
                "section": r["section"],
                "content": r["content"]
            }
            for r in results
        ]
    })
@app.route("/api/rag/qa", methods=["POST"])
def rag_qa():
    """
    Full RAG endpoint:
    - returns AI-generated answer + sources
    """
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    answer = rag_answer_with_llm(query, top_k=3)
    sources = rag_search(query, top_k=3)

    return jsonify({
        "query": query,
        "answer": answer,
        "sources": sources
    })


@app.route("/legal-assistant")
def legal_assistant_page():
    # expects templates/legal_assistant.html
    return render_template("legal_assistant.html")


@app.route("/legal_assistant.html")
def legal_assistant_html_alias():
    # support direct /legal_assistant.html
    return render_template("legal_assistant.html")
@app.route("/documents")
def documents_page():
    # UI page
    return render_template("documents.html")
@app.route("/api/documents/generate", methods=["POST"])
def generate_document():
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401

        data = request.get_json() or {}
        doc_type = data.get("doc_type")

        user = db.session.get(User, user_id)
        startup = Startup.query.filter_by(user_id=user_id).first()

        if not user or not startup:
            return jsonify({"error": "User or startup not found"}), 404

        if doc_type == "nda":
            file_path = generate_nda(
                call_llm,
                user,
                startup,
                data.get("other_party"),
                data.get("purpose")
            )

        elif doc_type == "mou":
            file_path = generate_mou(
                call_llm,
                user,
                startup,
                data.get("partner_name"),
                data.get("purpose")
            )

        elif doc_type == "rti":
            file_path = generate_rti(
                call_llm,
                user,
                startup,
                data.get("authority"),
                data.get("subject"),
                data.get("purpose")
            )

        elif doc_type == "pitch_deck":
            file_path = generate_pitch_deck(call_llm, user, startup)

        else:
            return jsonify({"error": "Invalid document type"}), 400

        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        return send_file(file_path, as_attachment=True)

    except Exception as e:
        logger.exception("Document generation failed")
        return jsonify({"error": str(e)}), 500

@app.route("/documents.html")
def documents_html_alias():
    return render_template("documents.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
