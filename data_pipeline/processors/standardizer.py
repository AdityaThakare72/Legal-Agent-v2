"""Standardize raw legal data (CSV / JSON / HuggingFace) into schema v2 chunks.

The public entry point is :func:`standardize_act`, which takes one registry
entry, loads its raw source in whatever format it happens to be, chunks the
text with the legal-aware splitter, and emits a list of dicts that match the
schema-v2 fields:

    chunk_id, chunk_text, source, category, title, act_name,
    section_number, chunk_type

Chunk IDs are deterministic: ``{act_slug}_sec_{section}_chunk_{i}`` where
``section`` is the section/article number when known, otherwise a short stable
hash of the record title (falling back to the record index).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Iterable

import pandas as pd

from config import LEGAL_CATEGORIES
from data_pipeline.processors.chunker import get_legal_chunker

logger = logging.getLogger(__name__)

# Persistent act_name -> LEGAL_CATEGORY lookup, produced offline by
# act_category_mapper.py. Consulted when a source carries no category, so that
# rebuilds reproduce the corrected categories without re-running the mapper.
_ACT_CATEGORY_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "act_category_map.json"
)


@lru_cache(maxsize=1)
def _act_category_map() -> dict:
    """Load the act->category map once, if present."""
    if not os.path.exists(_ACT_CATEGORY_MAP_PATH):
        return {}
    try:
        with open(_ACT_CATEGORY_MAP_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read %s; proceeding without it.", _ACT_CATEGORY_MAP_PATH)
        return {}

# Map a scraped source's own taxonomy (e.g. Nyaaya's top-level sections) onto
# our LEGAL_CATEGORIES. Anything not mapped and not already a valid category
# falls back to the registry-level category.
SCRAPED_CATEGORY_MAP = {
    "Crimes & Violence": "Criminal Law",
    "Police & Courts": "Criminal Law",
    "Money & Property": "Property & Real Estate",
    "Family & Marriage": "Family Law",
    "Labour & Employment": "Labour & Employment",
    "Health & Environment": "Environmental Law",
    "Government & Elections": "Constitutional Law",
    "Citizen Rights & Constitution": "Constitutional Law",
    "Media & Intellectual Property": "Cyber & IT Law",
}


def _map_category(raw: str | None) -> str | None:
    """Normalize a scraped category to a LEGAL_CATEGORY, or None if unknown."""
    if not raw:
        return None
    raw = str(raw).strip()
    if raw in SCRAPED_CATEGORY_MAP:
        return SCRAPED_CATEGORY_MAP[raw]
    if raw in LEGAL_CATEGORIES:
        return raw
    return None

# Candidate column names when auto-detecting fields in arbitrary tabular data.
_TEXT_COLUMNS = [
    "chunk_text", "text", "content", "description", "body",
    "act_text", "section_text", "answer", "output",
]
_TITLE_COLUMNS = [
    "title", "section_title", "heading", "name", "question", "instruction",
]
_SECTION_COLUMNS = ["section_number", "section", "sec", "article", "article_number"]
_SOURCE_COLUMNS = ["source", "url", "link", "source_url"]
# Multi-act datasets carry a per-row act name / category; when present these
# override the registry-level defaults so metadata stays granular.
_ACT_COLUMNS = ["act_name", "act_title", "act", "statute", "law_name"]
_CATEGORY_COLUMNS = ["category", "law_category", "domain"]

# Drop chunks shorter than this; they are almost always noise fragments
# ("2A", "Repealed ibid.") that pollute retrieval and waste embeddings.
MIN_CHUNK_CHARS = 25

# Stray leading punctuation left by splitting on ". " / ", " (plus the overlap),
# e.g. ". 25,000, and the registration...". Stripped from the front of chunks.
# Note: "(" is deliberately excluded so legal enumerations like "(1)" survive.
_LEADING_JUNK_RE = re.compile(r"^[\s.,;:!?•·\-–—]+")


def _clean_chunk_text(text: str) -> str:
    """Trim whitespace and stray leading sentence punctuation from a chunk."""
    return _LEADING_JUNK_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Intermediate record loading (one dict per raw source record, pre-chunk).
# Keys: title, text, section_number (str|None), source (str|None).
# ---------------------------------------------------------------------------
def _first_present(row: dict, candidates: Iterable[str]) -> Any:
    """Return the first candidate key present and non-empty in ``row``."""
    for key in candidates:
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return row[key]
    return None


def _extract_section(title: str | None, explicit: Any) -> str | None:
    """Derive a section/article number from an explicit field or a title."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    if title:
        # e.g. "Section 302 - Murder" or "Article 21." -> "302" / "21"
        match = re.search(r"(?:section|article)\s+(\d+[A-Za-z\-]*)", title, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _load_kaggle_json(path: str) -> list[dict]:
    """Load the Kaggle bare-act JSON (list of {title, description})."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    records: list[dict] = []
    for item in data:
        title = (item.get("title") or "").strip()
        text = (item.get("description") or item.get("text") or "").strip()
        if not text:
            continue
        records.append(
            {
                "title": title,
                "text": text,
                "section_number": _extract_section(title, item.get("section_number")),
                "source": None,
            }
        )
    return records


def _load_scraped_csv(path: str) -> list[dict]:
    """Load a scraped explainer CSV (title, url, content)."""
    df = pd.read_csv(path)
    records: list[dict] = []
    for row in df.to_dict(orient="records"):
        text = _first_present(row, _TEXT_COLUMNS)
        if not text:
            continue
        title = _first_present(row, _TITLE_COLUMNS) or ""
        records.append(
            {
                "title": str(title).strip(),
                "text": str(text).strip(),
                "section_number": _extract_section(str(title), _first_present(row, _SECTION_COLUMNS)),
                "source": _first_present(row, _SOURCE_COLUMNS),
                # Map the scraper's own category onto a LEGAL_CATEGORY when possible.
                "category": _map_category(_first_present(row, _CATEGORY_COLUMNS)),
            }
        )
    return records


def _load_generic_tabular(path: str) -> list[dict]:
    """Load an arbitrary CSV/JSON (HuggingFace export, manual CSV) by guessing columns."""
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        df = pd.DataFrame(raw if isinstance(raw, list) else raw.get("data", raw))
    else:
        df = pd.read_csv(path)

    records: list[dict] = []
    for row in df.to_dict(orient="records"):
        text = _first_present(row, _TEXT_COLUMNS)
        if not text:
            continue
        title = _first_present(row, _TITLE_COLUMNS) or ""
        section = _first_present(row, _SECTION_COLUMNS)
        records.append(
            {
                "title": str(title).strip(),
                "text": str(text).strip(),
                "section_number": _extract_section(str(title), section),
                "source": _first_present(row, _SOURCE_COLUMNS),
                # Optional per-row overrides for multi-act datasets.
                "act_name": _first_present(row, _ACT_COLUMNS),
                "category": _first_present(row, _CATEGORY_COLUMNS),
            }
        )
    return records


_LOADERS = {
    "kaggle_json": _load_kaggle_json,
    "scraped": _load_scraped_csv,
    "huggingface": _load_generic_tabular,
    "manual_csv": _load_generic_tabular,
}


def _load_raw(source_type: str, source_file: str) -> list[dict]:
    """Dispatch to the right loader for a source type."""
    loader = _LOADERS.get(source_type)
    if loader is None:
        raise ValueError(f"Unknown source_type '{source_type}'.")
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")
    records = loader(source_file)
    logger.info("Loaded %d raw records from %s (%s).", len(records), source_file, source_type)
    return records


# ---------------------------------------------------------------------------
# Deterministic chunk-id helper.
# ---------------------------------------------------------------------------
def _section_token(record: dict, index: int) -> str:
    """Stable section token for a record's chunk_id.

    Prefers the real section number; otherwise a short hash of the title; and
    finally the record index. For multi-act datasets a short hash of the row's
    act name is prefixed, so that "Section 1" of two different acts do not
    collide while ids stay deterministic (content-derived, not order-derived).
    """
    act = record.get("act_name")
    prefix = "a" + hashlib.md5(str(act).encode("utf-8")).hexdigest()[:6] + "_" if act else ""

    section = record.get("section_number")
    if section:
        return prefix + re.sub(r"\s+", "", str(section))
    title = record.get("title") or ""
    if title:
        return prefix + "h" + hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    return prefix + f"i{index}"


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def standardize_act(act: dict) -> list[dict]:
    """Standardize a single registry act into schema-v2 chunk dicts.

    Args:
        act: One entry from ``registry.yaml`` (slug, name, category,
            source_type, source_file, chunk_type, ...).

    Returns:
        A list of dicts, one per chunk, matching schema v2.
    """
    slug: str = act["slug"]
    act_name: str = act["name"]
    category: str = act["category"]
    source_type: str = act["source_type"]
    source_file: str = act["source_file"]
    chunk_type: str = act.get("chunk_type") or ("explainer" if source_type == "scraped" else "bare_act")
    default_source: str = act.get("source") or act_name

    records = _load_raw(source_type, source_file)
    chunker = get_legal_chunker()

    chunks: list[dict] = []
    seen_ids: set[str] = set()
    for r_index, record in enumerate(records):
        section_token = _section_token(record, r_index)
        # Per-row overrides (multi-act datasets) fall back to registry defaults.
        row_act = record.get("act_name") or act_name
        row_category = record.get("category") or category
        # When still uncategorised, consult the offline act->category map so
        # rebuilds reproduce the corrected categories automatically.
        if row_category == "General":
            row_category = _act_category_map().get(row_act, "General")
        row_section = record.get("section_number") or ""
        row_title = record.get("title") or (
            f"{row_act}, Section {row_section}" if row_section else row_act
        )
        pieces = chunker.split_text(record["text"])
        for i, piece in enumerate(pieces):
            piece = _clean_chunk_text(piece)
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunk_id = f"{slug}_sec_{section_token}_chunk_{i}"
            # Guard against title-hash collisions across different records.
            if chunk_id in seen_ids:
                chunk_id = f"{slug}_sec_{section_token}_r{r_index}_chunk_{i}"
            seen_ids.add(chunk_id)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_text": piece,
                    "source": record.get("source") or default_source,
                    "category": row_category,
                    "title": row_title,
                    "act_name": row_act,
                    "section_number": row_section,
                    "chunk_type": chunk_type,
                }
            )

    logger.info("Standardized '%s' into %d chunks.", slug, len(chunks))
    return chunks
