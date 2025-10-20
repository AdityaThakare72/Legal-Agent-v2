"""Download and standardize Indian-law datasets from HuggingFace.

Two responsibilities:

1. ``--inspect``: load each candidate dataset, print its columns / size / a
   sample row so a human can judge whether it holds real legal text (as opposed
   to bare Q&A pairs or metadata).

2. ``--save <key>``: write a chosen dataset to ``data_pipeline/data/raw/<key>.csv``
   with columns the standardizer can auto-detect (``title``, ``text``,
   ``section_number``, ``source``). Downstream, register the CSV in
   ``registry.yaml`` (source_type: huggingface) and run
   ``build_database.py --act <slug>``.

Usage:
    python data_pipeline/scrapers/hf_dataset_loader.py --inspect
    python data_pipeline/scrapers/hf_dataset_loader.py --save indian_law
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

RAW_DIR = os.path.join(_REPO_ROOT, "data_pipeline", "data", "raw")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("hf_loader")

# Candidate datasets to evaluate. `key` is the local short name used for the CSV
# filename and registry slug.
CANDIDATES: dict[str, str] = {
    "indian_law": "viber1/indian-law-dataset",
    "indian_laws": "mratanusarkar/Indian-Laws",
    "legal_dataset_india": "ShreyasP123/Legal-Dataset-for-india",
}

# Column-name candidates for mapping arbitrary datasets to our schema.
_TEXT_COLS = ["text", "content", "answer", "output", "response", "act_text",
              "section_text", "description", "body", "completion", "law"]
_TITLE_COLS = ["title", "act_title", "instruction", "question", "prompt",
               "heading", "name", "act", "section_title"]
_SECTION_COLS = ["section", "section_number", "article", "sec"]
_SOURCE_COLS = ["source", "url", "link", "citation"]
_ACT_COLS = ["act_name", "act_title", "act", "statute", "law_name"]


def _load(dataset_id: str) -> "pd.DataFrame":
    """Load a HuggingFace dataset (train split, or first available) as a DataFrame."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    split = "train" if "train" in ds else next(iter(ds.keys()))
    return ds[split].to_pandas()


def inspect_all() -> None:
    """Print structure and a sample row for each candidate dataset."""
    for key, dataset_id in CANDIDATES.items():
        print("\n" + "=" * 70)
        print(f"{key}  <-  {dataset_id}")
        print("=" * 70)
        try:
            df = _load(dataset_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED to load: {exc}")
            continue
        print(f"  rows: {len(df)}   columns: {list(df.columns)}")
        for col in df.columns:
            sample = str(df[col].iloc[0]) if len(df) else ""
            avg_len = int(df[col].astype(str).str.len().mean()) if len(df) else 0
            print(f"    - {col!r}: avg_len={avg_len}  sample={sample[:160]!r}")


def _pick(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate column present (case-insensitive)."""
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def save(key: str) -> str:
    """Download a candidate and save it as a standardizer-friendly CSV.

    Returns:
        The output CSV path.
    """
    if key not in CANDIDATES:
        raise SystemExit(f"Unknown dataset key '{key}'. Choices: {list(CANDIDATES)}")
    dataset_id = CANDIDATES[key]
    df = _load(dataset_id)
    cols = list(df.columns)

    text_col = _pick(cols, _TEXT_COLS)
    act_col = _pick(cols, _ACT_COLS)
    title_col = _pick(cols, _TITLE_COLS)
    section_col = _pick(cols, _SECTION_COLS)
    source_col = _pick(cols, _SOURCE_COLS)
    if text_col is None:
        raise SystemExit(f"No usable text column found in {dataset_id} (columns: {cols}).")

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    if act_col:  # per-row act name (multi-act datasets)
        out["act_name"] = df[act_col].astype(str)
    if title_col and title_col != act_col:  # a genuine title distinct from the act
        out["title"] = df[title_col].astype(str)
    if section_col:
        out["section_number"] = df[section_col].astype(str)
    out["source"] = df[source_col].astype(str) if source_col else dataset_id

    # Drop empty / trivially short text rows.
    out = out[out["text"].str.strip().str.len() >= 25].reset_index(drop=True)

    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{key}.csv")
    out.to_csv(path, index=False)
    logger.info("Saved %d rows to %s (text<-%s, title<-%s).", len(out), path, text_col, title_col)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect / download HuggingFace Indian-law datasets.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect", action="store_true", help="Inspect all candidate datasets.")
    group.add_argument("--save", metavar="KEY", help=f"Save one dataset. Choices: {list(CANDIDATES)}")
    args = parser.parse_args()

    if args.inspect:
        inspect_all()
    else:
        save(args.save)


if __name__ == "__main__":
    main()
