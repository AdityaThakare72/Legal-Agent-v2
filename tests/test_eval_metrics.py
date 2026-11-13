"""Offline tests for eval/metrics.py."""

import pytest

from eval.metrics import (
    first_hit_rank,
    hit_at_k,
    is_match,
    mrr_at_k,
    normalise_section,
    summarise,
    wrong_act_rate,
)

BNS = "Bharatiya Nyaya Sanhita, 2023"
EXPLAINER = "https://nyaaya.org/legal-explainer/how-to-file-an-fir/"


def result(act="", section="", source=""):
    return {"act_name": act, "section_number": section, "source": source}


SECTION_Q = {"gold": [{"act_name": BNS, "section_number": "303"}], "gold_source": None}
EXPLAINER_Q = {"gold": [], "gold_source": EXPLAINER}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("303", "303"),
        ("303(2)", "303"),
        (" 303 (2)(a) ", "303"),
        ("Section 303", "303"),
        ("sec. 303", "303"),
        ("s. 66 A", "66a"),
        ("498A", "498a"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalise_section(raw, expected):
    assert normalise_section(raw) == expected


def test_section_match_needs_act_and_section():
    assert is_match(result(BNS, "303"), SECTION_Q)
    assert is_match(result(BNS, "303(2)"), SECTION_Q)
    assert not is_match(result(BNS, "304"), SECTION_Q)
    assert not is_match(result("Motor Vehicle Act, 1988", "303"), SECTION_Q)


def test_empty_section_never_matches():
    question = {"gold": [{"act_name": BNS, "section_number": ""}], "gold_source": None}
    assert not is_match(result(BNS, ""), question)


def test_explainer_matches_on_source():
    assert is_match(result("Nyaaya Legal Explainers", "", EXPLAINER), EXPLAINER_Q)
    assert is_match(result("Nyaaya Legal Explainers", "", EXPLAINER.rstrip("/")), EXPLAINER_Q)
    assert not is_match(result("Nyaaya Legal Explainers", "", EXPLAINER + "x/"), EXPLAINER_Q)
    assert not is_match(result(BNS, "303"), EXPLAINER_Q)


def test_any_gold_entry_counts():
    question = {
        "gold": [
            {"act_name": "Code of Criminal Procedure Act, 1973", "section_number": "438"},
            {"act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023", "section_number": "482"},
        ],
        "gold_source": None,
    }
    assert is_match(result("Bharatiya Nagarik Suraksha Sanhita, 2023", "482"), question)
    assert is_match(result("Code of Criminal Procedure Act, 1973", "438"), question)


def test_first_hit_rank():
    results = [result(BNS, "1"), result(BNS, "303"), result(BNS, "303")]
    assert first_hit_rank(results, SECTION_Q) == 2
    assert first_hit_rank(results[:1], SECTION_Q) is None
    assert first_hit_rank([], SECTION_Q) is None


def test_hit_at_k_and_mrr():
    ranks = [1, 3, None, 10]
    assert hit_at_k(ranks, 1) == 0.25
    assert hit_at_k(ranks, 3) == 0.5
    assert hit_at_k(ranks, 5) == 0.5
    assert hit_at_k(ranks, 10) == 0.75
    assert mrr_at_k(ranks, 10) == pytest.approx((1 + 1 / 3 + 0 + 1 / 10) / 4)
    # A match below k counts as a miss.
    assert mrr_at_k([11], 10) == 0.0
    assert hit_at_k([], 5) == 0.0
    assert mrr_at_k([], 10) == 0.0


def test_wrong_act_rate():
    lists = [
        [result("Motor Vehicle Act, 1988"), result(BNS, "303")],
        [result(BNS, "1"), result("Motor Vehicles Act, 1988", "185")],
    ]
    # Only the mislabelled name counts, not the real Motor Vehicles Act.
    assert wrong_act_rate(lists) == 0.25
    assert wrong_act_rate([]) == 0.0


def test_summarise_keys():
    summary = summarise([1, None], [[result(BNS, "303")], []])
    assert summary["n"] == 2
    assert set(summary) == {"n", "hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "wrong_act_rate"}
    assert summary["hit@1"] == 0.5
