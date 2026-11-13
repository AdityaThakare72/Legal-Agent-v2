"""Retrieval metrics for the eval set: hit@k, MRR@k and the wrong-act rate.

A result is a dict with at least act_name, section_number and source.
A question is a row from eval/questions.jsonl.
"""

from __future__ import annotations

import re
from statistics import mean

KS = (1, 3, 5, 10)

# Chunks under this name are other acts filed under the wrong label (see
# config.EXCLUDED_ACTS). Any of them in the results is a wrong-act result.
MISLABELLED_ACTS = ["Motor Vehicle Act, 1988"]


def normalise_section(section: str | None) -> str:
    """Bring a section reference to the form stored in metadata.

    "Section 303(2)" -> "303", "s. 66 A" -> "66a", " 498A " -> "498a".
    """
    text = (section or "").strip().lower()
    text = re.sub(r"^(section|sec\.?|s\.)\s*", "", text)
    text = text.split("(", 1)[0]
    return re.sub(r"\s+", "", text)


def _same_source(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and a.rstrip("/") == b.rstrip("/")


def is_match(result: dict, question: dict) -> bool:
    """True when a result is one of the question's gold answers."""
    if _same_source(result.get("source"), question.get("gold_source")):
        return True
    section = normalise_section(result.get("section_number"))
    if not section:
        return False
    return any(
        result.get("act_name") == gold["act_name"] and section == normalise_section(gold["section_number"])
        for gold in question.get("gold", [])
    )


def first_hit_rank(results: list[dict], question: dict) -> int | None:
    """1-based rank of the first matching result, or None."""
    for rank, result in enumerate(results, start=1):
        if is_match(result, question):
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    """Share of questions with a match in the top k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at_k(ranks: list[int | None], k: int = 10) -> float:
    """Mean reciprocal rank, counting only matches in the top k."""
    if not ranks:
        return 0.0
    return mean(1 / r if r is not None and r <= k else 0.0 for r in ranks)


def wrong_act_rate(result_lists: list[list[dict]], mislabelled: list[str] = MISLABELLED_ACTS) -> float:
    """Share of all returned chunks whose act_name is in the mislabelled set."""
    total = sum(len(results) for results in result_lists)
    if not total:
        return 0.0
    wrong = sum(1 for results in result_lists for r in results if r.get("act_name") in mislabelled)
    return wrong / total


def summarise(ranks: list[int | None], result_lists: list[list[dict]]) -> dict:
    """All metrics for one group of questions."""
    summary = {"n": len(ranks)}
    for k in KS:
        summary[f"hit@{k}"] = hit_at_k(ranks, k)
    summary["mrr@10"] = mrr_at_k(ranks, 10)
    summary["wrong_act_rate"] = wrong_act_rate(result_lists)
    return summary
