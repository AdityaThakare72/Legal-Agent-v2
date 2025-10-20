"""Metadata-only patch: re-categorise 'General' chunks using the act->category map.

Reads ``data_pipeline/act_category_map.json`` (produced by
``processors/act_category_mapper.py``) and, for every chunk currently in the
"General" category whose act maps to a specific category, updates ONLY that
chunk's metadata via ``collection.update``. No embeddings are recomputed and no
embedding API is called — this touches metadata exclusively.

Usage:
    python data_pipeline/scripts/fix_categories.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chromadb  # noqa: E402

from config import COLLECTION_NAME, DB_PATH  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fix_categories")

MAP_PATH = os.path.join(_REPO_ROOT, "data_pipeline", "act_category_map.json")
READ_PAGE = 10000
UPDATE_BATCH = 500  # keep well under SQLite's bound-variable limit


def load_map() -> dict:
    """Load the act_name -> category lookup."""
    if not os.path.exists(MAP_PATH):
        raise SystemExit(f"Mapping not found: {MAP_PATH}. Run act_category_mapper.py first.")
    with open(MAP_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    category_map = load_map()
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    logger.info("Collection '%s' has %d chunks.", COLLECTION_NAME, total)

    pending_ids: list[str] = []
    pending_metas: list[dict] = []
    updated = 0

    def flush() -> None:
        nonlocal updated
        if not pending_ids:
            return
        collection.update(ids=pending_ids, metadatas=pending_metas)
        updated += len(pending_ids)
        logger.info("  updated %d chunks so far", updated)
        pending_ids.clear()
        pending_metas.clear()

    for offset in range(0, total, READ_PAGE):
        data = collection.get(include=["metadatas"], limit=READ_PAGE, offset=offset)
        for chunk_id, meta in zip(data["ids"], data["metadatas"]):
            if meta.get("category") != "General":
                continue
            new_cat = category_map.get(meta.get("act_name", ""))
            if not new_cat or new_cat == "General":
                continue
            updated_meta = dict(meta)
            updated_meta["category"] = new_cat
            pending_ids.append(chunk_id)
            pending_metas.append(updated_meta)
            if len(pending_ids) >= UPDATE_BATCH:
                flush()
    flush()

    logger.info("Re-categorised %d chunks (metadata-only, no re-embedding).", updated)

    # Fresh breakdown.
    by_category: Counter = Counter()
    for offset in range(0, total, READ_PAGE):
        data = collection.get(include=["metadatas"], limit=READ_PAGE, offset=offset)
        for meta in data["metadatas"]:
            by_category[meta.get("category", "?")] += 1

    print("\n" + "=" * 60)
    print(f"Re-categorised {updated} chunks. New category breakdown:")
    for name, n in by_category.most_common():
        print(f"  {n:>6}  {name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
