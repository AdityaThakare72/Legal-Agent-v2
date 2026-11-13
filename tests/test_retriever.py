"""Tests for the BM25 tokenizer in src/retriever.py."""

from src.retriever import tokenize


def test_punctuation_and_case_are_dropped():
    assert tokenize("Theft.—(1)") == ["theft", "1"]


def test_section_lookup_keeps_section_and_number():
    assert tokenize("Section 303 of BNS?") == ["section", "303", "bns"]


def test_lettered_section_stays_one_token():
    assert tokenize("498A") == ["498a"]


def test_stopwords_are_removed():
    assert tokenize("What is the punishment for theft under my lease?") == ["punishment", "theft", "lease"]


def test_query_and_chunk_tokens_line_up():
    chunk = "303. Theft.—(1) Whoever, intending to take dishonestly"
    assert {"303", "theft"} <= set(tokenize(chunk))
    assert set(tokenize("section 303 theft")) - {"section"} <= set(tokenize(chunk))
