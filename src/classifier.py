"""Query classifier — extracts metadata and builds a ChromaDB filter.

A cheap LLM (Gemini Flash-Lite) extracts the legal category, any referenced act
name, and any section number from the user's question, together with a
confidence signal. The returned dict is usable directly as a ChromaDB ``where``
filter.

Design decisions (see the Phase 3 spec):
  * The filter narrows by ``category`` only. Act name / section number are
    extracted for logging and transparency but are NOT used as equality filters,
    because the LLM's guessed act string rarely matches the stored ``act_name``
    verbatim and would zero out otherwise-good results.
  * Filtering is applied ONLY when the classifier is confident about a specific
    category. Vague queries ("what are my rights?") return an empty filter, so
    the retriever searches the whole corpus.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from config import LEGAL_CATEGORIES
from src.prompts import CLASSIFIER_PROMPT

logger = logging.getLogger(__name__)


def _parse_json_object(text: str) -> dict:
    """Extract a JSON object from a possibly fenced/preambled LLM response."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def classify_query(query: str, llm: Any) -> dict:
    """Classify a query and return a ChromaDB ``where`` filter.

    Args:
        query: The (already contextualized) user question.
        llm: A cheap LLM instance (Gemini Flash-Lite) for classification.

    Returns:
        A dict usable as a ChromaDB ``where`` filter. ``{"category": <cat>}``
        when the classifier is confident about a specific category; otherwise an
        empty dict ``{}`` meaning "no filter — search everything".
    """
    try:
        response = llm.invoke(CLASSIFIER_PROMPT.format(
            categories=", ".join(LEGAL_CATEGORIES),
            query=query,
        ))
    except Exception:
        logger.exception("Classifier LLM call failed; searching without a filter.")
        return {}

    data = _parse_json_object(getattr(response, "content", "") or "")
    if not data:
        logger.warning("Classifier returned unparseable output; no filter applied.")
        return {}

    category = data.get("category")
    act_name = data.get("act_name")
    section_number = data.get("section_number")
    confidence = str(data.get("confidence", "low")).lower()

    logger.info(
        "Classified query=%r -> category=%r act=%r section=%r confidence=%r",
        query, category, act_name, section_number, confidence,
    )

    # Only narrow by category when confident about a specific (non-General) one.
    if confidence == "high" and category in LEGAL_CATEGORIES and category != "General":
        logger.info("Applying metadata filter: category=%r", category)
        return {"category": category}

    logger.info("No confident specific category; searching all chunks (no filter).")
    return {}
