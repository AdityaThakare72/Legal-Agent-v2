"""Central configuration for Legal Agent v2.

All constants live here: model names, paths, prompts references, temperatures,
k values, and category lists. No magic strings elsewhere in the codebase.
"""

from __future__ import annotations

import os

# --- Secrets ---------------------------------------------------------------
# API key from Streamlit Cloud secrets in production, fallback to env for local
# dev (via a .env file loaded by python-dotenv).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv optional at runtime
    pass

try:
    import streamlit as st  # type: ignore

    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
except Exception:  # pragma: no cover - not running under Streamlit
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Models ----------------------------------------------------------------
GEMINI_CHAT_MODEL = "gemini-2.5-flash"            # Main generation model
GEMINI_LITE_MODEL = "gemini-2.5-flash-lite"       # Classifier + reranker (cheap)
# text-embedding-004 was retired from the Gemini API; gemini-embedding-001 is
# its supported successor. The DB is embedded at EMBEDDING_DIM below.
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768             # gemini-embedding-001 supports Matryoshka dims

# --- Generation ------------------------------------------------------------
CHAT_TEMPERATURE = 0.2          # Low temp for legal precision
MAX_CHAT_HISTORY = 10           # Keep last 10 exchanges (20 messages)
RETRIEVER_K = 10                # Retrieve 10, rerank to 5
RERANKER_TOP_K = 5              # Final docs sent to generator

# --- Vector Store ----------------------------------------------------------
DB_PATH = "./chroma_db"
# New collection re-embedded with gemini-embedding-001. The original v1
# collection ("legal_docs") was embedded with the now-retired text-embedding-004.
COLLECTION_NAME = "legal_docs_v2"

# --- Retriever weights -----------------------------------------------------
# EnsembleRetriever weights: [vector, bm25]
ENSEMBLE_WEIGHTS = [0.5, 0.5]

# --- Chunking (used by data_pipeline only) ---------------------------------
CHUNK_SIZE = 1000               # characters (~250 tokens)
CHUNK_OVERLAP = 200             # 20% overlap

# --- Metadata --------------------------------------------------------------
LEGAL_CATEGORIES = [
    "Criminal Law",
    "Constitutional Law",
    "Consumer Protection",
    "Family Law",
    "Property & Real Estate",
    "Labour & Employment",
    "Cyber & IT Law",
    "Motor Vehicle & Transport",
    "Contract & Commercial",
    "Right to Information",
    "Environmental Law",
    "Tax Law",
    "General",
]
