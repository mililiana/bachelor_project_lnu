"""
Flask backend for the LPNU Assistant web widget.
Connects the chat interface to the RAG pipeline.
Usage:
    python app.py                          # uses Gemini pipeline (cloud)
    python app.py --pipeline lapa          # uses local Lapa LLM (requires GPU)
"""

import argparse
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

rag_system = None
lapa_llm = None
pipeline_mode = "gemini"


def init_gemini_pipeline():
    from src.generation.rag_pipeline_gemini import CompleteRAGSystem
    return CompleteRAGSystem(
        db_path="vector_db",
        collection_name="hybrid_collection",
    )


def init_lapa_pipeline():
    """
    Initialize Lapa LLM for local answer generation.
    Uses the Gemini pipeline for search/retrieval, but Lapa for generation.
    """
    from src.generation.rag_pipeline_gemini import CompleteRAGSystem
    from src.generation.lapa_llm import LapaLLM

    # Use Gemini pipeline for query analysis + retrieval
    rag = CompleteRAGSystem(
        db_path="vector_db",
        collection_name="hybrid_collection",
    )
    # Use Lapa for answer generation
    lapa = LapaLLM()
    return rag, lapa


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_query = data.get("user_query", "").strip()

    if not user_query:
        return jsonify({"answer": "Please provide a question.", "sources": []}), 400

    logger.info(f"Query: {user_query}")

    try:
        if pipeline_mode == "lapa":
            # Use Gemini for retrieval, Lapa for generation
            result = rag_system.query(user_query, return_sources=True)
            source_docs = result.get("sources", [])
            answer = lapa_llm.generate_answer(user_query, source_docs)
        else:
            # Full Gemini pipeline
            result = rag_system.query(user_query, return_sources=True)
            source_docs = result.get("sources", [])
            answer = result.get("answer", "")

        sources = []
        for doc in source_docs:
            sources.append({
                "title": doc.get("title", "Без назви"),
                "url": doc.get("source_url", "#"),
            })

        return jsonify({
            "answer": answer,
            "sources": sources,
        })

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({
            "answer": "Вибачте, сталася помилка при обробці запиту.",
            "sources": [],
        }), 500


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("web", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPNU Assistant backend")
    parser.add_argument(
        "--pipeline",
        choices=["gemini", "lapa"],
        default="gemini",
        help="RAG pipeline to use: gemini (cloud) or lapa (local GPU)",
    )
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    pipeline_mode = args.pipeline
    logger.info(f"Initializing {pipeline_mode} pipeline...")

    if pipeline_mode == "lapa":
        rag_system, lapa_llm = init_lapa_pipeline()
    else:
        rag_system = init_gemini_pipeline()

    logger.info(f"Starting server on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
