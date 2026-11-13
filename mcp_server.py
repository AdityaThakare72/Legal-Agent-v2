"""MCP server for Legal Agent v2.

Exposes the retriever the Streamlit app uses (BM25 + vector search fused with
Reciprocal Rank Fusion) as MCP tools, so any MCP client can search the
Indian-law knowledge base.

No LLM runs inside this server. The app's classifier and reranker are skipped
on purpose: the model on the client side picks the category filter and writes
the answer. The only API call made here is the Gemini embedding of the query.

Run with:  python mcp_server.py      (stdio transport)
"""

# Same SQLite shim as app.py: ChromaDB needs a newer sqlite3 than some systems ship.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import json
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from config import (
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_DIM,
    GEMINI_EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    LEGAL_CATEGORIES,
    RETRIEVER_K,
)
from src.retriever import build_hybrid_retriever, drop_excluded_acts
from src.utils import load_bm25_documents

# stdout carries the MCP protocol on stdio transport, so logs must go to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("legal-mcp")


class LegalIndex:
    """All chunks in memory plus the hybrid retriever built over them."""

    def __init__(self, vector_store: Any) -> None:
        self.documents: list[Document] = drop_excluded_acts(load_bm25_documents(vector_store))
        self.retriever = build_hybrid_retriever(vector_store, self.documents)

    def search(
        self,
        query: str,
        category: str | None = None,
        act_name: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Hybrid search, optionally restricted to a category and/or an act."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if category and category not in LEGAL_CATEGORIES:
            raise ValueError(
                f"Unknown category {category!r}. Valid: {', '.join(LEGAL_CATEGORIES)}"
            )
        top_k = max(1, min(top_k, RETRIEVER_K))

        wanted = {}
        if category:
            wanted["category"] = category
        if act_name:
            wanted["act_name"] = act_name

        documents = self.retriever.invoke(query, where=_chroma_filter(wanted))

        # The retriever applies the filter to the vector half only (BM25 always
        # searches everything). A filter passed to this tool is an explicit
        # request, so drop any fused result that does not match it.
        if wanted:
            documents = [
                doc
                for doc in documents
                if all(doc.metadata.get(key) == value for key, value in wanted.items())
            ]

        return [_to_result(doc) for doc in documents[:top_k]]

    def list_acts(
        self,
        category: str | None = None,
        name_contains: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Acts in the knowledge base with their chunk counts, largest first."""
        needle = (name_contains or "").lower()
        counts: dict[tuple[str, str], int] = {}
        for doc in self.documents:
            name = doc.metadata.get("act_name") or ""
            doc_category = doc.metadata.get("category") or ""
            if not name:
                continue
            if category and doc_category != category:
                continue
            if needle and needle not in name.lower():
                continue
            key = (name, doc_category)
            counts[key] = counts.get(key, 0) + 1

        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [
            {"act_name": name, "category": doc_category, "chunks": count}
            for (name, doc_category), count in ranked[: max(1, limit)]
        ]


def _chroma_filter(wanted: dict) -> dict | None:
    """Turn {field: value} into a ChromaDB where clause ($and for two or more)."""
    if not wanted:
        return None
    if len(wanted) == 1:
        return dict(wanted)
    return {"$and": [{key: value} for key, value in wanted.items()]}


def _to_result(doc: Document) -> dict:
    """One retrieved chunk as a plain dict the client can cite from."""
    meta = doc.metadata or {}
    return {
        "act_name": meta.get("act_name", ""),
        "section_number": meta.get("section_number", ""),
        "title": meta.get("title", ""),
        "category": meta.get("category", ""),
        "source": meta.get("source", ""),
        "text": doc.page_content,
    }


def load_vector_store() -> Any:
    """Open the ChromaDB collection with the same embedding setup as app.py."""
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not set (put it in .env).")
    # MCP clients start this script from their own working directory, so the
    # relative DB_PATH from config is resolved against this file's folder.
    db_path = Path(__file__).resolve().parent / DB_PATH
    if not db_path.exists():
        raise RuntimeError(f"ChromaDB not found at {db_path}.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query",
        output_dimensionality=EMBEDDING_DIM,
    )
    return Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# Built on the first tool call, not at startup: loading every chunk and building
# the BM25 index takes a while, and the client should not wait for it to connect.
_index: LegalIndex | None = None


def get_index() -> LegalIndex:
    global _index
    if _index is None:
        logger.info("Loading knowledge base and building BM25 index ...")
        try:
            _index = LegalIndex(load_vector_store())
        except Exception as exc:
            # Most common cause: the repo was cloned without Git LFS, so
            # chroma_db holds small pointer files, not the database.
            raise RuntimeError(
                f"Could not open the knowledge base: {exc}. "
                "If chroma_db was cloned without Git LFS, run `git lfs pull`."
            ) from exc
        logger.info("Ready: %d chunks.", len(_index.documents))
    return _index


def _run(action: Any) -> Any:
    """Run a tool body, turning expected failures into messages the client sees.

    The SDK hides unexpected exceptions behind a generic error. ToolError is the
    one whose message reaches the calling model, so it can fix its input or tell
    the user what is missing.
    """
    try:
        return action(get_index())
    except (ValueError, RuntimeError) as exc:
        raise ToolError(str(exc)) from exc


mcp = MCPServer(
    "indian-law",
    instructions=(
        "Search a knowledge base of Indian legal acts. Call search_indian_law "
        "with a plain-language question or a section reference, and cite the "
        "act_name and section_number of the results you use. This is legal "
        "information, not legal advice."
    ),
)


@mcp.tool()
def search_indian_law(
    query: str,
    category: str | None = None,
    act_name: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Search Indian legal acts with hybrid retrieval (BM25 + vector, RRF).

    Args:
        query: A question in plain language, or a reference like "Section 302 IPC".
        category: Optional. Restrict to one category from legal://categories.
        act_name: Optional. Restrict to one act; use the exact name from list_acts.
        top_k: Number of chunks to return (1 to 10).

    Returns:
        Chunks, most relevant first, each with act_name, section_number, title,
        category, source and text.
    """
    return _run(lambda index: index.search(query, category=category, act_name=act_name, top_k=top_k))


@mcp.tool()
def list_acts(
    category: str | None = None,
    name_contains: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List acts in the knowledge base, with how many chunks each has.

    Use this to find the exact act_name to pass to search_indian_law.

    Args:
        category: Optional. Only acts in this category.
        name_contains: Optional. Case-insensitive substring of the act name.
        limit: Maximum number of acts to return.
    """
    return _run(lambda index: index.list_acts(category=category, name_contains=name_contains, limit=limit))


@mcp.resource("legal://categories")
def categories() -> str:
    """The legal categories every chunk is tagged with."""
    return json.dumps(LEGAL_CATEGORIES)


if __name__ == "__main__":
    mcp.run()
