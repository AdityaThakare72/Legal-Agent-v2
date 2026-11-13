"""Run the retrieval eval over eval/questions.jsonl.

Uses the app's own retriever pieces (src/retriever.py) against the real
chroma_db. No classifier filter and no reranker: this measures retrieval only.

    python eval/run_eval.py --mode hybrid --include-excluded --label before_hybrid
    python eval/run_eval.py --mode hybrid --label after_hybrid

Query embeddings are cached in eval/.cache/ so reruns do not call Gemini again.
All embeddings are fetched before timing starts, so latency is retrieval only.
"""

# Same SQLite shim as app.py: ChromaDB needs a newer sqlite3 than some systems ship.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_DIM,
    EXCLUDED_ACTS,
    GEMINI_EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    RETRIEVER_K,
)
from eval.metrics import first_hit_rank, summarise  # noqa: E402
from src.retriever import build_hybrid_retriever  # noqa: E402
from src.utils import load_bm25_documents  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
CACHE = ROOT / "eval" / ".cache" / "query_embeddings.json"
RESULTS = ROOT / "eval" / "results"
MODES = {"hybrid": ("vector", "bm25"), "bm25": ("bm25",), "vector": ("vector",)}

logger = logging.getLogger("eval")


def is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(s in text for s in ("429", "resource_exhausted", "resourceexhausted", "rate limit", "quota"))


def with_backoff(call, *args, tries: int = 6, delay: float = 2.0):
    """Retry a call on rate-limit errors, doubling the wait each time."""
    for attempt in range(tries):
        try:
            return call(*args)
        except Exception as exc:
            if not is_rate_limit(exc) or attempt == tries - 1:
                raise
            logger.warning("Rate limited, retrying in %.0fs", delay)
            time.sleep(delay)
            delay *= 2


class CachedEmbeddings(Embeddings):
    """Wraps the Gemini embeddings with an on-disk cache of query vectors."""

    def __init__(self, inner: Embeddings, path: Path) -> None:
        self.inner = inner
        self.path = path
        self.cache: dict[str, list[float]] = json.loads(path.read_text()) if path.exists() else {}

    def _key(self, text: str) -> str:
        raw = f"{GEMINI_EMBEDDING_MODEL}|{EMBEDDING_DIM}|retrieval_query|{text}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        if key not in self.cache:
            self.cache[key] = with_backoff(self.inner.embed_query, text)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache))
        return self.cache[key]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def open_vector_store(embeddings: Embeddings):
    from langchain_community.vectorstores import Chroma

    return Chroma(
        persist_directory=str(ROOT / DB_PATH),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def gemini_embeddings() -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not GOOGLE_API_KEY:
        sys.exit("GOOGLE_API_KEY is not set (put it in .env).")
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
        output_dimensionality=EMBEDDING_DIM,
    )


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def as_result(doc) -> dict:
    meta = doc.metadata or {}
    return {
        "act_name": meta.get("act_name", ""),
        "section_number": meta.get("section_number", ""),
        "source": meta.get("source", ""),
        "title": meta.get("title", ""),
    }


def grouped(rows: list[dict], key) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: summarise([r["rank"] for r in rs], [r["results"] for r in rs]) for name, rs in sorted(groups.items())}


def print_table(title: str, table: dict) -> None:
    print(f"\n{title}")
    print(f"  {'':28} {'n':>3} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'hit@10':>6} {'mrr@10':>6} {'wrong':>6}")
    for name, s in table.items():
        print(
            f"  {name[:28]:28} {s['n']:>3} {s['hit@1']:>6.2f} {s['hit@3']:>6.2f} {s['hit@5']:>6.2f}"
            f" {s['hit@10']:>6.2f} {s['mrr@10']:>6.3f} {s['wrong_act_rate']:>6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--include-excluded", action="store_true", help="turn EXCLUDED_ACTS off for this run")
    parser.add_argument("--label", help="results file name (default: <before|after>_<mode>)")
    args = parser.parse_args()
    label = args.label or f"{'before' if args.include_excluded else 'after'}_{args.mode}"
    excluded = [] if args.include_excluded else list(EXCLUDED_ACTS)
    use = MODES[args.mode]
    logging.basicConfig(level=logging.WARNING)

    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    skipped = [q["id"] for q in questions if not q.get("verified")]
    questions = [q for q in questions if q.get("verified")]
    if skipped:
        print(f"Skipping {len(skipped)} unverified questions: {', '.join(skipped)}")

    embeddings = CachedEmbeddings(gemini_embeddings(), CACHE)
    if "vector" in use:
        for q in questions:
            embeddings.embed_query(q["question"])

    vector_store = open_vector_store(embeddings)
    print("Loading chunks and building the BM25 index ...")
    retriever = build_hybrid_retriever(vector_store, load_bm25_documents(vector_store), excluded_acts=excluded)

    rows = []
    for q in questions:
        start = time.perf_counter()
        docs = retriever.invoke(q["question"], use=use)
        latency_ms = (time.perf_counter() - start) * 1000
        if retriever.last_failures:
            sys.exit(f"{q['id']}: {', '.join(retriever.last_failures)} search failed; see the log above.")
        results = [as_result(doc) for doc in docs]
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "category": q["category"],
            "known_issue": q.get("known_issue"),
            "rank": first_hit_rank(results, q),
            "latency_ms": round(latency_ms, 2),
            "results": results,
        })

    clean = [r for r in rows if not r["known_issue"]]
    latencies = [r["latency_ms"] for r in rows]
    summary = {
        "all": summarise([r["rank"] for r in rows], [r["results"] for r in rows]),
        "without_known_issues": summarise([r["rank"] for r in clean], [r["results"] for r in clean]),
        "by_type": grouped(rows, lambda r: r["type"]),
        "by_category": grouped(rows, lambda r: r["category"]),
        "latency_ms": {"p50": round(percentile(latencies, 50), 2), "p95": round(percentile(latencies, 95), 2)},
    }
    report = {
        "label": label,
        "mode": args.mode,
        "excluded_acts": excluded,
        "top_k": RETRIEVER_K,
        "summary": summary,
        "questions": [
            {**{k: v for k, v in r.items() if k != "results"}, "top5": r["results"][:5]} for r in rows
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{label}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{label}: mode={args.mode}, excluded_acts={excluded}, top_k={RETRIEVER_K}")
    print_table("Overall", {"all": summary["all"], "without known issues": summary["without_known_issues"]})
    print_table("By type", summary["by_type"])
    print_table("By category", summary["by_category"])
    print(f"\nRetrieval latency: p50 {summary['latency_ms']['p50']:.1f} ms, p95 {summary['latency_ms']['p95']:.1f} ms")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
