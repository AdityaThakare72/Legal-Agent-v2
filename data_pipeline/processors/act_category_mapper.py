"""Classify act names into LEGAL_CATEGORIES with a cheap LLM, once, offline.

Many acts (notably the bulk HuggingFace bare-act corpus) land in the catch-all
"General" category because their source carries no category field. This script
reads the distinct ``act_name`` values that are currently categorised as
"General" from ChromaDB, asks Gemini Flash-Lite to map each to one of the
canonical ``LEGAL_CATEGORIES``, and persists the result to
``data_pipeline/act_category_map.json``.

The map is consumed by:
  * ``scripts/fix_categories.py`` — a metadata-only patch of the live DB, and
  * ``standardizer.py`` — so future rebuilds get the right category for free.

Usage:
    python data_pipeline/processors/act_category_mapper.py
    python data_pipeline/processors/act_category_mapper.py --batch-size 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chromadb  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402

from config import (  # noqa: E402
    COLLECTION_NAME,
    DB_PATH,
    GEMINI_LITE_MODEL,
    GOOGLE_API_KEY,
    LEGAL_CATEGORIES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("act_category_mapper")

MAP_PATH = os.path.join(_REPO_ROOT, "data_pipeline", "act_category_map.json")
DEFAULT_BATCH_SIZE = 100
BATCH_PAUSE_SECONDS = 1.0

PROMPT_TEMPLATE = (
    "You are classifying Indian legal acts into subject-matter categories.\n\n"
    "Allowed categories (use these EXACT strings, nothing else):\n{categories}\n\n"
    "For each act name below, choose the single best-fitting category. If you are "
    "genuinely unsure, use \"General\".\n\n"
    "Return ONLY a JSON object mapping each act name (verbatim) to its category, "
    "no markdown, no commentary. Example: "
    '{{"Some Act, 1999": "Tax Law", "Other Act, 2001": "Criminal Law"}}\n\n'
    "Act names:\n{acts}"
)


def get_general_act_names() -> list[str]:
    """Return the distinct act_name values currently categorised as General."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    acts: set[str] = set()
    for offset in range(0, total, 10000):
        data = collection.get(include=["metadatas"], limit=10000, offset=offset)
        for meta in data.get("metadatas") or []:
            if meta.get("category") == "General":
                name = meta.get("act_name")
                if name:
                    acts.add(name)
    logger.info("Found %d distinct act names in 'General'.", len(acts))
    return sorted(acts)


def _parse_json_object(text: str) -> dict:
    """Extract a JSON object from a possibly fenced/preambled LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM JSON response; skipping this batch.")
        return {}


def classify_batch(llm: ChatGoogleGenerativeAI, act_names: list[str]) -> dict:
    """Classify one batch of act names, returning {act_name: category}."""
    prompt = PROMPT_TEMPLATE.format(
        categories="\n".join(f"- {c}" for c in LEGAL_CATEGORIES),
        acts="\n".join(act_names),
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        logger.exception("LLM call failed for a batch; defaulting the batch to General.")
        return {}
    raw = _parse_json_object(getattr(response, "content", "") or "")
    # Keep only valid categories; anything else -> General.
    result: dict[str, str] = {}
    valid = set(LEGAL_CATEGORIES)
    for name in act_names:
        cat = raw.get(name)
        result[name] = cat if cat in valid else "General"
    return result


def build_map(batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Classify all General act names and persist the mapping to JSON."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set (check your .env).")

    act_names = get_general_act_names()
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_LITE_MODEL, temperature=0.0, google_api_key=GOOGLE_API_KEY
    )

    mapping: dict[str, str] = {}
    for start in range(0, len(act_names), batch_size):
        batch = act_names[start : start + batch_size]
        mapping.update(classify_batch(llm, batch))
        logger.info("  classified %d/%d", min(start + batch_size, len(act_names)), len(act_names))
        time.sleep(BATCH_PAUSE_SECONDS)

    with open(MAP_PATH, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)

    non_general = sum(1 for v in mapping.values() if v != "General")
    logger.info(
        "Wrote %d mappings to %s (%d mapped to a specific category, %d left General).",
        len(mapping), MAP_PATH, non_general, len(mapping) - non_general,
    )
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify General act names into LEGAL_CATEGORIES.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    build_map(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
