"""Utility helpers for Legal Agent v2.

Chat-history trimming, source formatting for the UI, and loading documents out
of ChromaDB to build the BM25 index.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from config import MAX_CHAT_HISTORY

logger = logging.getLogger(__name__)


def trim_chat_history(history: list[Any], max_exchanges: int = MAX_CHAT_HISTORY) -> list[Any]:
    """Keep only the last ``max_exchanges`` pairs of messages.

    Args:
        history: List of LangChain message objects (Human/AI alternating).
        max_exchanges: Number of Human+AI pairs to retain.

    Returns:
        The trimmed history (last ``max_exchanges * 2`` messages).
    """
    max_messages = max_exchanges * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def format_sources(documents: list[Document]) -> list[dict]:
    """Extract source metadata from documents for display in the UI.

    Args:
        documents: Retrieved LangChain ``Document`` objects.

    Returns:
        A list of dicts with the human-relevant metadata fields plus a short
        text snippet.
    """
    sources: list[dict] = []
    for doc in documents:
        meta = doc.metadata or {}
        snippet = doc.page_content or ""
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "…"
        sources.append(
            {
                "title": meta.get("title", "Untitled"),
                "act_name": meta.get("act_name", ""),
                "section_number": meta.get("section_number", ""),
                "category": meta.get("category", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "source": meta.get("source", ""),
                "snippet": snippet,
            }
        )
    return sources


def load_bm25_documents(vector_store: Any) -> list[Document]:
    """Load all documents from ChromaDB to seed the BM25 index.

    Args:
        vector_store: A LangChain Chroma vector store.

    Returns:
        A list of ``Document`` objects reconstructed from the collection.

    Raises:
        RuntimeError: If the collection cannot be read.
    """
    # Page through the collection — a single unpaginated get() over a large
    # collection exceeds SQLite's bound-variable limit.
    page = 5000
    documents: list[Document] = []
    try:
        total = vector_store._collection.count()
        for offset in range(0, total, page):
            data = vector_store.get(
                include=["documents", "metadatas"], limit=page, offset=offset
            )
            texts = data.get("documents") or []
            metadatas = data.get("metadatas") or []
            for text, metadata in zip(texts, metadatas):
                if not text:
                    continue
                documents.append(Document(page_content=text, metadata=metadata or {}))
    except Exception as exc:  # pragma: no cover - depends on live DB
        logger.exception("Failed to load documents from ChromaDB for BM25.")
        raise RuntimeError(f"Could not load documents for BM25 index: {exc}") from exc

    logger.info("Loaded %d documents for BM25 index.", len(documents))
    return documents
