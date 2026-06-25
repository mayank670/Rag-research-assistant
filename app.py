"""
RAG Research Assistant - Flask Application
"""

import os
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai

from utils.pdf_loader import process_pdf
from utils.embeddings import (
    get_chroma_client,
    get_or_create_collection,
    store_chunks,
    delete_source_documents,
    get_collection_stats,
)
from utils.summarizer import generate_summary
from graph.workflow import run_rag_pipeline

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "rag-research-assistant-secret-2024")

UPLOAD_FOLDER = BASE_DIR / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

UPLOAD_FOLDER.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ── Gemini Init ───────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set — set it in .env before using the app")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# ── ChromaDB ──────────────────────────────────────────────────────────────────
chroma_client = get_chroma_client(str(VECTORSTORE_DIR))
collection = get_or_create_collection(chroma_client)

# In-memory document registry
uploaded_docs: dict = {}


def rebuild_uploaded_docs_from_chromadb():
    """
    On startup, reconstruct uploaded_docs from ChromaDB metadata.
    This ensures documents uploaded in a previous session are still accessible
    without needing to re-upload them.
    """
    global uploaded_docs
    try:
        total = collection.count()
        if total == 0:
            return

        # Fetch all metadata (no embeddings/docs needed)
        results = collection.get(include=["metadatas"])
        if not results or not results.get("metadatas"):
            return

        seen = {}  # source → {pages: set, chunks: int}
        for meta in results["metadatas"]:
            source = meta.get("source")
            page = meta.get("page", 1)
            if not source:
                continue
            if source not in seen:
                seen[source] = {"pages": set(), "chunks": 0}
            seen[source]["pages"].add(page)
            seen[source]["chunks"] += 1

        for source, info in seen.items():
            file_path = UPLOAD_FOLDER / source
            uploaded_docs[source] = {
                "filename": source,
                "path": str(file_path),
                "chunks": info["chunks"],
                "pages": max(info["pages"]) if info["pages"] else 1,
            }

        logger.info(f"Restored {len(uploaded_docs)} document(s) from ChromaDB on startup")

    except Exception as e:
        logger.warning(f"Could not rebuild doc registry from ChromaDB: {e}")


# Rebuild on startup
rebuild_uploaded_docs_from_chromadb()


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def require_api_key():
    if not os.getenv("GOOGLE_API_KEY"):
        return jsonify({"error": "GOOGLE_API_KEY is not configured. Add it to your .env file."}), 503
    return None


def has_documents() -> bool:
    """Check both in-memory registry AND ChromaDB — whichever has data."""
    return bool(uploaded_docs) or collection.count() > 0


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_pdf():
    err = require_api_key()
    if err:
        return err

    if "files" not in request.files:
        return jsonify({"error": "No files provided."}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected."}), 400

    results = []

    for file in files:
        if not file or file.filename == "":
            continue

        if not allowed_file(file.filename):
            results.append({"filename": file.filename, "success": False, "error": "Only PDF files are accepted."})
            continue

        filename = secure_filename(file.filename)
        save_path = UPLOAD_FOLDER / filename

        try:
            file.save(str(save_path))
            logger.info(f"Saved: {save_path}")

            chunks = process_pdf(str(save_path), chunk_size=1000, chunk_overlap=200)

            # Remove old version if re-uploading
            if filename in uploaded_docs:
                delete_source_documents(collection, filename)

            num_stored = store_chunks(chunks, collection, filename)

            uploaded_docs[filename] = {
                "filename": filename,
                "path": str(save_path),
                "chunks": num_stored,
                "pages": max(c["page"] for c in chunks),
            }

            results.append({
                "filename": filename,
                "success": True,
                "chunks": num_stored,
                "pages": uploaded_docs[filename]["pages"],
            })

        except ValueError as e:
            results.append({"filename": filename, "success": False, "error": str(e)})
        except Exception as e:
            logger.error(f"Upload error for {filename}: {e}")
            results.append({"filename": filename, "success": False, "error": f"Processing failed: {str(e)}"})

    success_count = sum(1 for r in results if r.get("success"))

    return jsonify({
        "results": results,
        "documents": list(uploaded_docs.values()),
        "message": f"{success_count}/{len(results)} file(s) processed successfully.",
    })


@app.route("/api/documents", methods=["GET"])
def list_documents():
    stats = get_collection_stats(collection)
    return jsonify({
        "documents": list(uploaded_docs.values()),
        "total_chunks": stats["total_chunks"],
    })


@app.route("/api/documents/<filename>", methods=["DELETE"])
def delete_document(filename: str):
    if filename not in uploaded_docs:
        return jsonify({"error": "Document not found."}), 404

    try:
        delete_source_documents(collection, filename)
        file_path = Path(uploaded_docs[filename]["path"])
        if file_path.exists():
            file_path.unlink()
        del uploaded_docs[filename]
        return jsonify({
            "message": f"'{filename}' deleted successfully.",
            "documents": list(uploaded_docs.values()),
        })
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def ask_question():
    err = require_api_key()
    if err:
        return err

    data = request.get_json()
    question = (data or {}).get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    # ✅ FIX: check ChromaDB directly, not just in-memory dict
    if collection.count() == 0:
        return jsonify({"error": "No documents found. Please upload a PDF first."}), 400

    try:
        result = run_rag_pipeline(question, collection)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks_used": result["chunks_used"],
            "error": result.get("error"),
        })
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize_documents():
    err = require_api_key()
    if err:
        return err

    # ✅ FIX: check ChromaDB directly
    if collection.count() == 0:
        return jsonify({"error": "No documents found. Please upload a PDF first."}), 400

    # Use uploaded_docs names if available, else read from ChromaDB metadata
    if uploaded_docs:
        doc_names = list(uploaded_docs.keys())
    else:
        try:
            results = collection.get(include=["metadatas"])
            doc_names = list({m.get("source", "document") for m in results["metadatas"] if m.get("source")})
        except Exception:
            doc_names = ["uploaded document"]

    result = generate_summary(collection, doc_names)

    if "error" in result:
        return jsonify(result), 500

    return jsonify(result)


@app.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"message": "Chat cleared."})


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "ok",
        "api_key_set": bool(os.getenv("GOOGLE_API_KEY")),
        "documents_loaded": len(uploaded_docs),
        "total_chunks": collection.count(),
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔬 RAG Research Assistant")
    print("=" * 40)
    print(f"   Upload folder  : {UPLOAD_FOLDER}")
    print(f"   Vector store   : {VECTORSTORE_DIR}")
    print(f"   Docs in memory : {len(uploaded_docs)}")
    print(f"   Chunks in DB   : {collection.count()}")
    print(f"   API key set    : {'✅' if GOOGLE_API_KEY else '❌  Set GOOGLE_API_KEY in .env'}")
    print(f"   Running at     : http://localhost:5000")
    print("=" * 40 + "\n")
    app.run(debug=True, port=5000)
