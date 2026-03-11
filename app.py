import os
import joblib
import re
import logging
import tempfile
import json
import numpy as np
import pandas as pd

from mistralai import Mistral
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
from docx import Document
from fpdf import FPDF
from flask import send_from_directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, "generated_docs")
os.makedirs(EXPORT_DIR, exist_ok=True)

from models import db, User, Startup
from matcher import load_schemes, match_schemes

# --- NEW: imports for document summarizer ---
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

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
mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))


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
# --- Startup Success Predictor Model ---
try:
    startup_model = joblib.load("startup_xgb_model.pkl")
    logger.info("✅ Startup predictor model loaded")
except Exception as e:
    logger.exception("❌ Failed to load startup predictor model")
    startup_model = None

# --- NEW: Tesseract path (adjust if installed elsewhere) ---


# --- NEW: helper to extract text from PDFs / images ---
def extract_text_from_file(file_path, file_ext):
    text = ""
    try:
        if file_ext.lower() == ".pdf":
            pages = convert_from_path(file_path)
            # Convert PDF pages to images
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
# ---------------- RAG v2 INITIALIZATION ---------------- #



EMBED_MODEL = "all-MiniLM-L6-v2"

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Loading reranker...")

from urllib.parse import urlparse

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

url = urlparse(DATABASE_URL)

rag_conn = psycopg2.connect(
    host=url.hostname,
    database=url.path[1:],
    user=url.username,
    password=url.password,
    port=url.port
)
rag_cur = rag_conn.cursor()

def retrieve(query):
    query_embedding = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    rag_cur.execute("""
        SELECT doc_id, act_name, section, content
        FROM legal_docs
        ORDER BY embedding <=> %s::vector
        LIMIT 20;
    """, (query_embedding,))

    return rag_cur.fetchall()
# ✅ RETURN docs, not fetchall() again


def rag_answer_with_llm(query):
    retrieved = retrieve(query)

    if not retrieved:
        return "I could not find relevant legal information in the database.", [], None

    top_docs = retrieved[:5]

    context = ""
    for doc in top_docs:
        trimmed = doc[3][:800]  # truncate long sections
        context += f"""
Act: {doc[1]}
Section: {doc[2]}

{trimmed}
"""


    # -------- Draft Generation --------
    draft_prompt = f"""
You are a legal assistant for startups in India.

Rules:
- Answer directly and concisely.
- Use ONLY the provided context.
- Do NOT invent information.
- Cite sections clearly: [Source: Section <number>]

Question:
{query}

Context:
{context}

Answer:
"""

    draft_response = mistral_client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {"role": "system", "content": "You are a legal assistant for startups in India."},
        {"role": "user", "content": draft_prompt}
    ]
)


    draft_answer = draft_response.choices[0].message.content.strip()

    # -------- Evaluation --------
    evaluation = evaluate_answer_with_mistral(
        query,
        context,
        draft_answer
    )

    final_answer = draft_answer


    sources = [
        {
            "act": doc[1],
            "section": doc[2],
            "content": doc[3] 
        }
        for doc in top_docs
    ]

    return final_answer, sources, evaluation


def call_llm(messages, model="mistral-small-latest"):
    response = mistral_client.chat.complete(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content.strip()

def evaluate_answer_with_mistral(query, context, draft_answer):
    evaluation_prompt = f"""
You are a strict legal evaluator for Indian startup law.

Evaluate the answer based on:

1. Grounding: Is every statement supported by the context?
2. Relevance: Does it directly answer the question?
3. Conciseness: Is it precise and not overly verbose?
4. Legal correctness: Is the interpretation accurate?

Return JSON ONLY in this format:

{{
"score": <number between 1 and 10>,
"hallucination": true/false,
"needs_refinement": true/false,
"reason": "brief explanation"
}}

Question:
{query}

Context:
{context}

Answer:
{draft_answer}
"""

    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "You are a strict legal evaluator."},
            {"role": "user", "content": evaluation_prompt}
        ]
    )

    try:
        result = json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "score": 5,
            "hallucination": True,
            "needs_refinement": True,
            "reason": "Evaluator formatting error"
        }

    return result


def refine_answer_with_mistral(query, context, draft_answer):
    refinement_prompt = f"""
You are refining a legal answer for Indian startup law.

Fix the answer by:
- Removing unsupported or hallucinated claims
- Making it concise and precise
- Ensuring it directly answers the question
- Ensuring citations follow this format:
[Source: Section <number>]

Return ONLY the final improved answer.

Question:
{query}

Context:
{context}

Original Answer:
{draft_answer}

Final Answer:
"""

    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "You refine legal answers."},
            {"role": "user", "content": refinement_prompt}
        ]
    )

    return response.choices[0].message.content.strip()

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
@app.route("/signup.html")
def signup_html():
    return render_template("signup.html")

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
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    retrieved = retrieve(query)
    top_docs = retrieved[:5]

    return jsonify({
        "results": [
            {
                "act": doc[1],
                "section": doc[2],
                "content": doc[3][:600]
            }
            for doc in top_docs
        ]
    })
@app.route("/api/rag/qa", methods=["POST"])
def rag_qa():
    data = request.get_json() or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query is required"}), 400

    answer, sources, evaluation = rag_answer_with_llm(query)

    return jsonify({
        "query": query,
        "answer": answer,
        "sources": sources,
        "evaluation": evaluation
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

    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.get_json() or {}
    doc_type = data.get("doc_type")

    if not doc_type:
        return jsonify({"error": "Document type missing"}), 400

    user = db.session.get(User, user_id)
    startup = Startup.query.filter_by(user_id=user_id).first()

    draft_text = None
    file_path = None

    if doc_type == "nda":
        file_path, draft_text = generate_nda(
            call_llm, user, startup,
            data.get("other_party"),
            data.get("purpose"),
            preview=True
        )

    elif doc_type == "mou":
        file_path, draft_text = generate_mou(
            call_llm, user, startup,
            data.get("partner_name"),
            data.get("purpose"),
            preview=True
        )

    elif doc_type == "rti":
        file_path, draft_text = generate_rti(
            call_llm, user, startup,
            data.get("authority"),
            data.get("subject"),
            data.get("purpose"),
            preview=True
        )

    elif doc_type == "pitch_deck":
        file_path, draft_text = generate_pitch_deck(
            call_llm, startup,
            preview=True
        )

    else:
        return jsonify({"error": "Invalid document type"}), 400

    return jsonify({
        "draft": draft_text,
        "file_path": file_path
    })
@app.route("/documents.html")
def documents_html_alias():
    return render_template("documents.html")


EXPORT_DIR = os.path.join(BASE_DIR, "generated_docs")
os.makedirs(EXPORT_DIR, exist_ok=True)


@app.route("/api/documents/export", methods=["POST"])
def export_document():

    data = request.get_json()
    text = data.get("text")
    format_type = data.get("format","docx")

    if not text:
        return jsonify({"error":"No text provided"}),400

    file_name = f"Edited_Draft.{format_type}"
    file_path = os.path.join(EXPORT_DIR, file_name)

    # ---------- DOCX ----------
    if format_type == "docx":

        doc = Document()
        for line in text.split("\n"):
            doc.add_paragraph(line)

        doc.save(file_path)

    # ---------- PDF ----------
    elif format_type == "pdf":

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        for line in text.split("\n"):
            pdf.multi_cell(0,8,line)

        pdf.output(file_path)

    else:
        return jsonify({"error":"Invalid format"}),400

    return jsonify({
        "file_url": f"/download/{file_name}"
    })
@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(
        "generated_docs",
        filename,
        as_attachment=True
    )
@app.route("/debug/sample")
def sample_row():
    rag_cur.execute("""
        SELECT doc_id, act_name, section, LEFT(content, 300)
        FROM legal_docs
        LIMIT 5;
    """)
    rows = rag_cur.fetchall()
    return jsonify(rows)

@app.route("/api/startup/predict", methods=["POST"])
def startup_predict():

    if not startup_model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json() or {}

    try:
        age = float(data.get("age_startup_year", 0))
        funding = float(data.get("funding_total_usd", 0))
        rounds = float(data.get("funding_rounds", 0))
        milestones = float(data.get("milestones", 0))
        participants = float(data.get("avg_participants", 0))
        relationships = float(data.get("relationships", 0))
        category = data.get("category", "other")

        has_VC = int(data.get("has_VC", 0))
        has_angel = int(data.get("has_angel", 0))
        has_Seed = int(data.get("has_Seed", 0))

        # ---- Tier Engineering ----
        if relationships <= 5:
            tier = 4
        elif relationships <= 10:
            tier = 3
        elif relationships <= 16:
            tier = 2
        else:
            tier = 1

        # ---- Category Encoding ----
        category_flags = {
    "is_software": 1 if category == "software" else 0,
    "is_web": 1 if category == "web" else 0,
    "is_mobile": 1 if category == "mobile" else 0,
    "is_enterprise": 1 if category == "enterprise" else 0,
    "is_advertising": 1 if category == "advertising" else 0,
    "is_gamesvideo": 1 if category == "gamesvideo" else 0,
    "is_ecommerce": 1 if category == "ecommerce" else 0,
    "is_biotech": 1 if category == "biotech" else 0,
    "is_consulting": 1 if category == "consulting" else 0,
    "is_othercategory": 1 if category == "other" else 0
}


        # ---- Feature Dictionary ----
        features_dict = {
            'age_first_funding_year': 0,
            'age_last_funding_year': 0,
            'age_first_milestone_year': 0,
            'age_last_milestone_year': 0,
            'funding_rounds': rounds,
            'funding_total_usd': funding,
            'milestones': milestones,

            'is_CA': 0,
            'is_NY': 0,
            'is_MA': 0,
            'is_TX': 0,
            'is_otherstate': 1,

            **category_flags,

            'has_VC': has_VC,
            'has_angel': has_angel,
            'has_roundA': 0,
            'has_roundB': 0,
            'has_roundC': 0,
            'has_roundD': 0,

            'avg_participants': participants,
            'is_top500': 0,
            'has_RoundABCD': 0,
            'has_Investor': 0,
            'has_Seed': has_Seed,
            'invalid_startup': 0,

            'age_startup_year': age,
            'tier_relationships': tier
        }

        features_df = pd.DataFrame([features_dict])

        prediction = startup_model.predict(features_df)[0]
        probability = startup_model.predict_proba(features_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "success_probability": round(float(probability) * 100, 2),
            "verdict": "Likely to Succeed 🚀" if prediction == 1 else "Likely to Fail ⚠️"
        })

    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500
@app.route("/predictor")
@app.route("/predictor.html")
def predictor_page():
    return render_template("predictor.html")




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)