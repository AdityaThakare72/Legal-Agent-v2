"""Fetch the 2023 Indian criminal codes (BNS / BNSS / BSA) from HuggingFace.

Source: ``GSMS-B/indian-legal-sections-bns-bnss-bsa-2023`` — 1,059 sections of
the three codes that replaced the IPC, CrPC, and Indian Evidence Act on
1 July 2024. This script splits that single dataset into three per-act CSVs in
``data_pipeline/data/raw/`` with columns the standardizer auto-detects
(``text``, ``act_name``, ``title``, ``section_number``, ``source``), using the
full statutory names.

Usage:
    python data_pipeline/scrapers/fetch_bns_codes.py
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(_REPO_ROOT, "data_pipeline", "data", "raw")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fetch_bns_codes")

DATASET_ID = "GSMS-B/indian-legal-sections-bns-bnss-bsa-2023"

# dataset 'act' value -> (output slug, full statutory act_name)
ACTS = {
    "BNS 2023": ("bns_2023", "Bharatiya Nyaya Sanhita, 2023"),
    "BNSS 2023": ("bnss_2023", "Bharatiya Nagarik Suraksha Sanhita, 2023"),
    "BSA 2023": ("bsa_2023", "Bharatiya Sakshya Adhiniyam, 2023"),
}


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    split = "train" if "train" in ds else next(iter(ds.keys()))
    df = ds[split].to_pandas()
    logger.info("Loaded %d rows from %s.", len(df), DATASET_ID)

    os.makedirs(RAW_DIR, exist_ok=True)
    for act_value, (slug, full_name) in ACTS.items():
        sub = df[df["act"] == act_value]
        if sub.empty:
            logger.warning("No rows for act=%r; skipping.", act_value)
            continue
        out = pd.DataFrame(
            {
                "text": sub["text"].astype(str),
                "act_name": full_name,
                "title": sub["section_title"].astype(str),
                "section_number": sub["section_number"].astype(str),
                "source": sub["source_label"].astype(str),
            }
        )
        path = os.path.join(RAW_DIR, f"{slug}.csv")
        out.to_csv(path, index=False)
        logger.info("Wrote %d sections -> %s (%s).", len(out), path, full_name)


if __name__ == "__main__":
    main()
