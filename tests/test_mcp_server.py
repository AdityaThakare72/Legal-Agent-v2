"""Tests for the MCP server.

They run against a tiny ChromaDB built in a temp folder with fake embeddings,
so they need no API key, no network and not the real knowledge base.
"""

import asyncio
import json

import pytest
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

import mcp_server
from mcp_server import LegalIndex

EXCLUDED = "Motor Vehicle Act, 1988"

CHUNKS = [
    ("Punishment for murder is death or imprisonment for life.", "Bharatiya Nyaya Sanhita, 2023", "103", "Criminal Law"),
    ("Theft is dishonestly taking movable property out of possession.", "Bharatiya Nyaya Sanhita, 2023", "303", "Criminal Law"),
    ("Cheating by personation is punishable with imprisonment.", "Bharatiya Nyaya Sanhita, 2023", "319", "Criminal Law"),
    ("A consumer may file a complaint about a defective product.", "Consumer Protection Act, 2019", "35", "Consumer Protection"),
    ("Product liability action lies against the product manufacturer.", "Consumer Protection Act, 2019", "84", "Consumer Protection"),
    ("Driving without a licence is punishable with a fine.", "Motor Vehicles Act, 1988", "181", "Motor Vehicle & Transport"),
    # Mislabelled chunks like the ones in the real DB: other acts filed under
    # the excluded name. They must never come back.
    ("Whoever commits theft shall be punished with imprisonment.", EXCLUDED, "379", "Motor Vehicle & Transport"),
    ("Punishment for theft in a dwelling house.", EXCLUDED, "380", "Motor Vehicle & Transport"),
]


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    documents = [
        Document(
            page_content=text,
            metadata={
                "act_name": act,
                "section_number": section,
                "category": category,
                "title": f"Section {section}",
                "source": "test",
            },
        )
        for text, act, section, category in CHUNKS
    ]
    vector_store = Chroma.from_documents(
        documents,
        DeterministicFakeEmbedding(size=32),
        persist_directory=str(tmp_path_factory.mktemp("chroma")),
        collection_name="test",
    )
    return LegalIndex(vector_store)


def call_search(arguments):
    """Call search_indian_law over the MCP protocol and return the result."""

    async def run():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool("search_indian_law", arguments)

    return asyncio.run(run())


def test_search_returns_citable_chunks(index):
    results = index.search("punishment for murder", top_k=3)

    assert 1 <= len(results) <= 3
    assert set(results[0]) == {"act_name", "section_number", "title", "category", "source", "text"}
    # The embeddings here are fake (random), so only the BM25 half is meaningful.
    # Being ranked first by BM25 is enough to stay in the fused top 3.
    assert "103" in [r["section_number"] for r in results]


def test_category_filter_is_strict(index):
    results = index.search("punishment", category="Consumer Protection")

    assert results
    assert all(r["category"] == "Consumer Protection" for r in results)


def test_category_and_act_filter_together(index):
    results = index.search(
        "theft of property",
        category="Criminal Law",
        act_name="Bharatiya Nyaya Sanhita, 2023",
    )

    assert results
    assert all(r["act_name"] == "Bharatiya Nyaya Sanhita, 2023" for r in results)


def test_bad_input_is_rejected(index):
    with pytest.raises(ValueError):
        index.search("   ")
    with pytest.raises(ValueError):
        index.search("theft", category="Space Law")


def test_top_k_is_clamped(index):
    assert len(index.search("punishment", top_k=0)) == 1
    assert len(index.search("punishment", top_k=999)) <= 10


def test_list_acts_counts_and_filters(index):
    acts = index.list_acts()
    assert acts[0] == {"act_name": "Bharatiya Nyaya Sanhita, 2023", "category": "Criminal Law", "chunks": 3}
    assert len(acts) == 3

    assert [a["act_name"] for a in index.list_acts(name_contains="motor")] == ["Motor Vehicles Act, 1988"]
    assert [a["chunks"] for a in index.list_acts(category="Consumer Protection")] == [2]


def test_tools_work_over_the_mcp_protocol(index, monkeypatch):
    """Connect a real MCP client to the server in-process and call the tools."""
    monkeypatch.setattr(mcp_server, "_index", index)

    async def run():
        async with Client(mcp_server.mcp) as client:
            tools = await client.list_tools()
            search = await client.call_tool(
                "search_indian_law", {"query": "defective product complaint", "top_k": 2}
            )
            bad = await client.call_tool(
                "search_indian_law", {"query": "theft", "category": "Space Law"}
            )
            resource = await client.read_resource("legal://categories")
            return tools, search, bad, resource

    tools, search, bad, resource = asyncio.run(run())

    assert {tool.name for tool in tools.tools} == {"search_indian_law", "list_acts"}
    assert not search.is_error
    assert bad.is_error
    # The calling model must see why it failed, not a generic error.
    assert "Unknown category" in bad.content[0].text
    assert "Criminal Law" in json.loads(resource.contents[0].text)


def test_excluded_act_never_returned(index):
    for query in ["punishment for theft", "theft in a dwelling house", "imprisonment"]:
        results = index.search(query, top_k=10)
        assert results
        assert EXCLUDED not in [r["act_name"] for r in results]

    # The vector half on its own also leaves them out.
    vector_docs = index.retriever._vector_search("theft", None)
    assert vector_docs
    assert EXCLUDED not in [d.metadata["act_name"] for d in vector_docs]


def test_excluded_act_not_listed(index):
    names = [a["act_name"] for a in index.list_acts(limit=100)]
    assert EXCLUDED not in names
    assert index.list_acts(name_contains="motor vehicle act") == []


def test_filter_combined_with_exclusion(index):
    # Both acts share this category, so only the exclusion separates them.
    results = index.search("punishment for theft", category="Motor Vehicle & Transport")

    assert [r["act_name"] for r in results] == ["Motor Vehicles Act, 1988"]
    vector_docs = index.retriever._vector_search("theft", {"category": "Motor Vehicle & Transport"})
    assert [d.metadata["act_name"] for d in vector_docs] == ["Motor Vehicles Act, 1988"]


def test_search_has_no_warning_when_all_is_well(index, monkeypatch):
    monkeypatch.setattr(mcp_server, "_index", index)

    result = call_search({"query": "theft", "top_k": 3})

    assert not result.is_error
    body = json.loads(result.content[0].text)
    assert body["warning"] is None
    assert body["results"]


def test_failing_vector_search_gives_a_warning(index, monkeypatch):
    class BrokenStore:
        def similarity_search(self, *args, **kwargs):
            raise RuntimeError("API key not valid")

    monkeypatch.setattr(mcp_server, "_index", index)
    monkeypatch.setattr(index.retriever, "_vector_store", BrokenStore())

    result = call_search({"query": "theft of property", "top_k": 3})

    assert not result.is_error
    body = json.loads(result.content[0].text)
    assert body["results"]
    assert "keyword-only" in body["warning"]
    assert index.retriever.last_failures == ["vector"]


def test_startup_check_rejects_bad_api_key(monkeypatch):
    class BrokenEmbeddings:
        def embed_query(self, text):
            raise RuntimeError("API key not valid")

    class Store:
        embeddings = BrokenEmbeddings()

    monkeypatch.setattr(mcp_server, "_index", None)
    monkeypatch.setattr(mcp_server, "load_vector_store", lambda: Store())

    with pytest.raises(ToolError, match="Gemini API key was rejected"):
        mcp_server.get_index()
    assert mcp_server._index is None

    # Over the protocol the client sees the same message.
    result = call_search({"query": "theft"})
    assert result.is_error
    assert "Gemini API key was rejected" in result.content[0].text
