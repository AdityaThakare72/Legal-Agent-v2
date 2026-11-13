"""Write eval/REVIEW.md: every question with the text of its gold chunks.

Reads chroma_db directly (no embeddings, no API key) and fails if a gold
label does not exist in the database.

Run from the repo root:  python eval/make_review.py
"""

import json
import re
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import COLLECTION_NAME, DB_PATH  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
REVIEW = ROOT / "eval" / "REVIEW.md"
SNIPPET = 200

# Explainer pages end with a rating widget and comments; skip those chunks
# when picking text to show.
BOILERPLATE = re.compile(r"Vote count|Leave a Reply")


def chunk_order(chunk_id: str) -> int:
    match = re.search(r"chunk_(\d+)$", chunk_id)
    return int(match.group(1)) if match else 0


def fetch(collection, where: dict) -> list[str]:
    data = collection.get(where=where, include=["documents"])
    pairs = sorted(zip(data["ids"], data["documents"]), key=lambda p: chunk_order(p[0]))
    return [text for _, text in pairs]


def snippet(texts: list[str]) -> str:
    text = " ".join(texts)
    # BNS/BNSS/BSA chunks start with a "[Context: ...]" line that only repeats the title.
    text = re.sub(r"\[Context:[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:SNIPPET].replace("|", "\\|")


def main() -> None:
    client = chromadb.PersistentClient(path=str(ROOT / DB_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    rows = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    lines = [
        "# Eval questions: review",
        "",
        f"{len(rows)} questions. Gold text is the first {SNIPPET} characters of the gold "
        "section as stored in chroma_db. Regenerate with `python eval/make_review.py`.",
        "",
        "| id | type | question | gold | gold text |",
        "|---|---|---|---|---|",
    ]
    missing = []
    for row in rows:
        golds, texts = [], []
        if row.get("gold_source"):
            chunks = fetch(collection, {"source": row["gold_source"]})
            chunks = [t for t in chunks if not BOILERPLATE.search(t)] or chunks
            golds.append(row["gold_source"].replace("https://nyaaya.org/legal-explainer/", "nyaaya: "))
            texts.append(snippet(chunks[:1]))
            if not chunks:
                missing.append(f"{row['id']}: {row['gold_source']}")
        for gold in row["gold"]:
            chunks = fetch(
                collection,
                {"$and": [{"act_name": gold["act_name"]}, {"section_number": gold["section_number"]}]},
            )
            golds.append(f"{gold['act_name']} s.{gold['section_number']}")
            texts.append(snippet(chunks))
            if not chunks:
                missing.append(f"{row['id']}: {gold['act_name']} s.{gold['section_number']}")
        question = row["question"].replace("|", "\\|")
        if row.get("note"):
            question += f"<br>*Note: {row['note']}*".replace("|", "\\|")
        if row.get("known_issue"):
            question += f"<br>**Known issue:** {row['known_issue']}".replace("|", "\\|")
        lines.append(
            f"| {row['id']} | {row['type']} | {question} | {'<br>'.join(golds)} | {'<br><br>'.join(texts)} |"
        )

    if missing:
        sys.exit("Gold labels not found in chroma_db:\n" + "\n".join(missing))
    REVIEW.write_text("\n".join(lines) + "\n")
    print(f"Wrote {REVIEW.relative_to(ROOT)} ({len(rows)} questions).")


if __name__ == "__main__":
    main()
