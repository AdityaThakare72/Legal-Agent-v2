"""Summary table across eval runs, and the worst questions of one run.

    python eval/compare.py before_hybrid after_hybrid after_bm25 after_vector
    python eval/compare.py after_hybrid --worst 5
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


def load(label: str) -> dict:
    return json.loads((RESULTS / f"{label}.json").read_text())


def table(labels: list[str], subset: str) -> str:
    lines = [
        "| run | hit@1 | hit@5 | MRR@10 | wrong-act rate | p50 latency |",
        "|---|---|---|---|---|---|",
    ]
    for label in labels:
        report = load(label)
        s = report["summary"][subset]
        p50 = report["summary"]["latency_ms"]["p50"]
        lines.append(
            f"| {label} | {s['hit@1']:.2f} | {s['hit@5']:.2f} | {s['mrr@10']:.3f}"
            f" | {s['wrong_act_rate']:.1%} | {p50:.0f} ms |"
        )
    return "\n".join(lines)


def worst(label: str, n: int) -> str:
    questions = {
        q["id"]: q for q in (json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip())
    }
    rows = load(label)["questions"]
    # Misses first, then the lowest-ranked hits.
    rows = sorted(rows, key=lambda r: (-(r["rank"] or 99), r["id"]))[:n]
    out = []
    for r in rows:
        q = questions[r["id"]]
        gold = q["gold_source"] or "; ".join(f"{g['act_name']} s.{g['section_number']}" for g in q["gold"])
        rank = r["rank"] if r["rank"] else "miss"
        out.append(f"{r['id']} ({q['type']}, rank {rank}): {q['question']}")
        out.append(f"  gold: {gold}")
        for i, res in enumerate(r["top5"][:3], start=1):
            name = res["act_name"] + (f" s.{res['section_number']}" if res["section_number"] else f" ({res['title']})")
            out.append(f"  got {i}: {name}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--worst", type=int, help="list the N worst questions of the first run")
    parser.add_argument("--subset", default="all", choices=["all", "without_known_issues"])
    args = parser.parse_args()
    if args.worst:
        print(worst(args.labels[0], args.worst))
    else:
        print(table(args.labels, args.subset))


if __name__ == "__main__":
    main()
