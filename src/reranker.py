"""LLM-based reranker.

Takes the hybrid retriever's output and asks a cheap LLM (Gemini Flash-Lite) to
reorder the documents by relevance to the query, returning the top ``top_k``.
Any failure (LLM error, unparseable output) degrades gracefully to the original
ordering — this step must never crash the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.documents import Document

from config import RERANKER_TOP_K
from src.prompts import RERANKER_PROMPT

logger = logging.getLogger(__name__)

# Characters of each document shown to the reranker — enough to judge relevance
# while keeping the prompt small and cheap.
EXCERPT_CHARS = 300


def _parse_index_array(text: str, n: int) -> list[int]:
    """Parse a JSON array of document indices, keeping only valid, unique ones."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0].strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    order: list[int] = []
    seen: set[int] = set()
    for item in raw:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)
    return order


def rerank_documents(
    query: str,
    documents: list[Document],
    llm: Any,
    top_k: int = RERANKER_TOP_K,
) -> list[Document]:
    """Rerank retrieved documents by LLM-judged relevance to the query.

    Args:
        query: The user question.
        documents: Documents from the hybrid retriever (typically 10).
        llm: A cheap LLM instance (Gemini Flash-Lite) for scoring.
        top_k: Number of documents to return.

    Returns:
        The ``top_k`` most relevant documents in reranked order. On any failure,
        the original first ``top_k`` documents are returned unchanged.
    """
    if not documents:
        return []
    if len(documents) <= 1:
        return documents[:top_k]

    numbered = "\n\n".join(
        f"[{i}] {doc.page_content[:EXCERPT_CHARS]}" for i, doc in enumerate(documents)
    )
    try:
        response = llm.invoke(RERANKER_PROMPT.format(query=query, documents=numbered))
    except Exception:
        logger.exception("Reranker LLM call failed; falling back to original order.")
        return documents[:top_k]

    order = _parse_index_array(getattr(response, "content", "") or "", len(documents))
    if not order:
        logger.warning("Reranker returned no usable ordering; keeping original order.")
        return documents[:top_k]

    # Append any indices the LLM omitted, preserving original order, so we never
    # drop documents just because the model forgot one.
    for i in range(len(documents)):
        if i not in order:
            order.append(i)

    reranked = [documents[i] for i in order]
    logger.info("Reranked %d docs; top order=%s", len(documents), order[:top_k])
    return reranked[:top_k]
