"""Master CLI to build / update the ChromaDB knowledge base.

Modes:
    --rebuild-all      Drop the collection and re-process every active act.
    --act <slug>       Process (upsert) a single act from the registry.
    --changed-only     Process acts whose last_updated is newer than the last build.

Embeds with gemini-embedding-001 (task_type="retrieval_document") at the
configured dimensionality, in batches of 100 with a 1-second pause between
batches to stay under the rate limit.

Run from the repo root, e.g.:
    python data_pipeline/build_database.py --rebuild-all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import date, datetime

import yaml

# Ensure the repo root is importable when run as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chromadb  # noqa: E402
from langchain_community.vectorstores import Chroma  # noqa: E402
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # noqa: E402

from config import (  # noqa: E402
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_DIM,
    GEMINI_EMBEDDING_MODEL,
    GOOGLE_API_KEY,
)
from data_pipeline.processors.standardizer import standardize_act  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("build_database")

REGISTRY_PATH = os.path.join(_REPO_ROOT, "data_pipeline", "registry.yaml")
BUILD_STATE_PATH = os.path.join(_REPO_ROOT, "data_pipeline", "build_state.json")
BATCH_SIZE = 100
BATCH_PAUSE_SECONDS = 1.0
MAX_RETRIES = 6


# ---------------------------------------------------------------------------
# Registry & build-state helpers.
# ---------------------------------------------------------------------------
def load_registry() -> list[dict]:
    """Load the act entries from registry.yaml."""
    with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("acts", []) or []


def load_build_state() -> dict:
    """Load the persisted build state (last-processed dates per act)."""
    if not os.path.exists(BUILD_STATE_PATH):
        return {"last_build": None, "acts": {}}
    with open(BUILD_STATE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_build_state(state: dict) -> None:
    """Persist the build state."""
    state["last_build"] = datetime.now().isoformat(timespec="seconds")
    with open(BUILD_STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _as_date(value) -> date | None:
    """Coerce a registry last_updated value into a date."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Embedding & storage.
# ---------------------------------------------------------------------------
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Build the document embedding function."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set (check your .env).")
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document",
        output_dimensionality=EMBEDDING_DIM,
    )


def get_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Open (or create) the Chroma collection."""
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def _sanitize(meta: dict) -> dict:
    """Coerce metadata values to Chroma-safe scalars (no None)."""
    clean: dict = {}
    for key, value in meta.items():
        clean[key] = "" if value is None else str(value)
    return clean


def embed_and_store(store: Chroma, chunks: list[dict]) -> int:
    """Embed chunk texts and upsert them into the collection in batches.

    Args:
        store: The Chroma vector store.
        chunks: Schema-v2 chunk dicts.

    Returns:
        The number of chunks written.
    """
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]
        metadatas = [
            _sanitize(
                {
                    "source": c["source"],
                    "category": c["category"],
                    "title": c["title"],
                    "act_name": c["act_name"],
                    "section_number": c["section_number"],
                    "chunk_type": c["chunk_type"],
                }
            )
            for c in batch
        ]
        ids = [c["chunk_id"] for c in batch]
        _add_with_retry(store, texts, metadatas, ids, start=start)
        logger.info("  embedded %d/%d", min(start + BATCH_SIZE, total), total)
        time.sleep(BATCH_PAUSE_SECONDS)
    return total


def _add_with_retry(store: Chroma, texts, metadatas, ids, *, start: int) -> None:
    """Upsert one batch, retrying transient errors (e.g. rate limits) with backoff."""
    delay = 5.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == MAX_RETRIES:
                logger.error("Batch at offset %d failed after %d attempts.", start, MAX_RETRIES)
                raise
            logger.warning(
                "Batch at offset %d failed (attempt %d/%d): %s. Retrying in %.0fs.",
                start, attempt, MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def delete_act_chunks(act_name: str) -> None:
    """Delete all existing chunks for an act, so it can be re-upserted cleanly."""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return  # collection doesn't exist yet
    collection.delete(where={"act_name": act_name})
    logger.info("Cleared existing chunks for act_name=%r.", act_name)


def drop_collection() -> None:
    """Delete the whole collection (used by --rebuild-all)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Dropped collection '%s'.", COLLECTION_NAME)
    except Exception:
        logger.info("Collection '%s' did not exist; nothing to drop.", COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def process_act(store: Chroma, act: dict, *, clear_first: bool) -> int:
    """Standardize, (optionally) clear, and store one act."""
    slug = act["slug"]
    logger.info("Processing act '%s' (%s) ...", slug, act.get("name"))
    chunks = standardize_act(act)
    if not chunks:
        logger.warning("Act '%s' produced no chunks; skipping.", slug)
        return 0
    if clear_first:
        delete_act_chunks(act["name"])
    written = embed_and_store(store, chunks)
    logger.info("Stored %d chunks for '%s'.", written, slug)
    return written


def run_rebuild_all() -> None:
    """Drop and rebuild the entire collection from all active acts."""
    acts = [a for a in load_registry() if a.get("status", "active") == "active"]
    logger.info("Rebuilding all: %d active act(s).", len(acts))
    drop_collection()
    store = get_vector_store(get_embeddings())

    state = {"last_build": None, "acts": {}}
    per_act: Counter = Counter()
    for act in acts:
        per_act[act["slug"]] = process_act(store, act, clear_first=False)
        state["acts"][act["slug"]] = str(act.get("last_updated", ""))

    save_build_state(state)
    _report(store, per_act)


def run_single_act(slug: str) -> None:
    """Process (upsert) a single act by slug."""
    acts = load_registry()
    match = next((a for a in acts if a["slug"] == slug), None)
    if match is None:
        raise SystemExit(f"No act with slug '{slug}' in the registry.")
    store = get_vector_store(get_embeddings())
    written = process_act(store, match, clear_first=True)

    state = load_build_state()
    state.setdefault("acts", {})[slug] = str(match.get("last_updated", ""))
    save_build_state(state)
    logger.info("Done. Upserted %d chunks for '%s'.", written, slug)
    _report(store)


def run_changed_only() -> None:
    """Process acts whose last_updated is newer than the recorded build."""
    acts = [a for a in load_registry() if a.get("status", "active") == "active"]
    state = load_build_state()
    recorded = state.get("acts", {})

    changed = []
    for act in acts:
        slug = act["slug"]
        current = _as_date(act.get("last_updated"))
        previous = _as_date(recorded.get(slug))
        if previous is None or (current and current > previous):
            changed.append(act)

    if not changed:
        logger.info("Nothing changed since the last build.")
        return

    logger.info("Changed acts: %s", ", ".join(a["slug"] for a in changed))
    store = get_vector_store(get_embeddings())
    for act in changed:
        process_act(store, act, clear_first=True)
        recorded[act["slug"]] = str(act.get("last_updated", ""))

    state["acts"] = recorded
    save_build_state(state)
    _report(store)


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def _report(store: Chroma, per_act: Counter | None = None) -> None:
    """Print final counts and category/act breakdowns."""
    collection = store._collection
    total = collection.count()
    print("\n" + "=" * 60)
    print(f"Collection '{COLLECTION_NAME}' now has {total} chunks.")

    # Page through metadatas — a single get() on a large collection exceeds
    # SQLite's bound-variable limit.
    by_category: Counter = Counter()
    by_act: Counter = Counter()
    page = 10000
    for offset in range(0, total, page):
        data = collection.get(include=["metadatas"], limit=page, offset=offset)
        for m in data.get("metadatas") or []:
            by_category[m.get("category", "?")] += 1
            by_act[m.get("act_name", "?")] += 1

    print("\nBy category:")
    for name, n in by_category.most_common():
        print(f"  {n:>6}  {name}")
    print("\nBy act:")
    for name, n in by_act.most_common():
        print(f"  {n:>6}  {name}")
    if per_act:
        print("\nThis run wrote:")
        for slug, n in per_act.items():
            print(f"  {n:>6}  {slug}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Legal Agent knowledge base.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rebuild-all", action="store_true", help="Drop and rebuild everything.")
    group.add_argument("--act", metavar="SLUG", help="Process/upsert a single act.")
    group.add_argument("--changed-only", action="store_true", help="Process acts changed since last build.")
    args = parser.parse_args()

    if args.rebuild_all:
        run_rebuild_all()
    elif args.act:
        run_single_act(args.act)
    elif args.changed_only:
        run_changed_only()


if __name__ == "__main__":
    main()
