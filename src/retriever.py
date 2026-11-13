"""Hybrid retriever: ChromaDB vector search + BM25 keyword search.

``langchain`` 1.x moved ``EnsembleRetriever`` into the ``langchain_classic``
compatibility shim, which this project forbids. We therefore implement the
ensemble directly using weighted Reciprocal Rank Fusion (RRF) over a Chroma
vector retriever and a BM25 keyword retriever — the same fusion strategy
``EnsembleRetriever`` uses.

The vector half accepts an optional ChromaDB ``where`` filter (supplied by the
classifier) to narrow semantic search by metadata; the BM25 keyword half always
searches the whole corpus, which keeps retrieval robust even when the filter is
too aggressive.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import ENSEMBLE_WEIGHTS, EXCLUDED_ACTS, RETRIEVER_K

logger = logging.getLogger(__name__)

# RRF smoothing constant (matches LangChain's EnsembleRetriever default).
RRF_C = 60


def drop_excluded_acts(
    documents: list[Document], excluded_acts: list[str] = EXCLUDED_ACTS
) -> list[Document]:
    """Remove chunks whose act_name is in ``excluded_acts``."""
    return [doc for doc in documents if doc.metadata.get("act_name") not in excluded_acts]


def with_exclusion(where: dict | None, excluded_acts: list[str] = EXCLUDED_ACTS) -> dict | None:
    """Add the excluded-acts condition to a ChromaDB ``where`` filter."""
    if not excluded_acts:
        return where
    exclude = {"act_name": {"$nin": list(excluded_acts)}}
    if not where:
        return exclude
    return {"$and": [where, exclude]}


class HybridRetriever:
    """Weighted RRF ensemble of a vector retriever and a BM25 retriever.

    Exposes ``.invoke(query)`` returning a fused, deduplicated list of
    ``Document`` objects, so it is a drop-in for the rest of the pipeline.
    """

    def __init__(
        self,
        vector_store: Any,
        bm25_retriever: Any,
        weights: list[float],
        top_k: int,
        excluded_acts: list[str] = EXCLUDED_ACTS,
    ) -> None:
        self._vector_store = vector_store
        self._excluded_acts = list(excluded_acts)
        self._bm25_retriever = bm25_retriever
        self._vector_weight, self._bm25_weight = weights
        self._top_k = top_k
        # Names of the sub-retrievers ("vector", "bm25") that failed on the
        # last invoke. Callers that want to warn about degraded results read it.
        self.last_failures: list[str] = []

    def _vector_search(self, query: str, where: dict | None) -> list[Document]:
        """Semantic search, optionally narrowed by a ChromaDB ``where`` filter."""
        where = with_exclusion(where, self._excluded_acts)
        kwargs: dict[str, Any] = {"k": self._top_k}
        if where:
            kwargs["filter"] = where
        return self._vector_store.similarity_search(query, **kwargs)

    def invoke(
        self,
        query: str,
        where: dict | None = None,
        use: tuple[str, ...] = ("vector", "bm25"),
    ) -> list[Document]:
        """Retrieve from the sub-retrievers and fuse via weighted RRF.

        Args:
            query: The search query.
            where: Optional ChromaDB metadata filter applied to the vector half
                only. BM25 always searches the entire corpus.
            use: Which sub-retrievers to run. Both by default; the eval runner
                passes one of them to measure it alone.

        Returns:
            The top-``top_k`` fused documents, most relevant first.
        """
        sources = [
            ("vector", lambda q: self._vector_search(q, where), self._vector_weight),
            ("bm25", self._bm25_retriever.invoke, self._bm25_weight),
        ]
        sources = [source for source in sources if source[0] in use]

        self.last_failures = []
        # doc key -> [document, fused_score]
        fused: dict[str, list[Any]] = {}
        for name, search, weight in sources:
            try:
                docs = search(query)
            except Exception:
                logger.exception("The %s retriever failed; skipping it for this query.", name)
                self.last_failures.append(name)
                continue
            for rank, doc in enumerate(docs, start=1):
                key = doc.page_content
                score = weight * (1.0 / (RRF_C + rank))
                if key in fused:
                    fused[key][1] += score
                else:
                    fused[key] = [doc, score]

        ranked = sorted(fused.values(), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[: self._top_k]]


def build_hybrid_retriever(
    vector_store: Any,
    chunks_for_bm25: list[Document],
    excluded_acts: list[str] = EXCLUDED_ACTS,
) -> HybridRetriever:
    """Build a hybrid retriever combining vector and keyword search.

    Args:
        vector_store: A LangChain Chroma vector store.
        chunks_for_bm25: All documents in the collection, used to build the
            in-memory BM25 index. Chunks of ``excluded_acts`` are dropped.
        excluded_acts: Acts kept out of both halves. Defaults to
            ``config.EXCLUDED_ACTS``; the eval passes [] for "before" numbers.

    Returns:
        A :class:`HybridRetriever` combining ChromaDB vector search and BM25
        with the weights from ``config.ENSEMBLE_WEIGHTS``.

    Raises:
        RuntimeError: If either retriever cannot be constructed.
    """
    chunks_for_bm25 = drop_excluded_acts(chunks_for_bm25, excluded_acts)
    if not chunks_for_bm25:
        raise RuntimeError("No documents supplied for the BM25 index.")

    try:
        bm25_retriever = BM25Retriever.from_documents(chunks_for_bm25)
        bm25_retriever.k = RETRIEVER_K
    except Exception as exc:
        logger.exception("Failed to build BM25 retriever.")
        raise RuntimeError(f"Could not build BM25 retriever: {exc}") from exc

    logger.info(
        "Hybrid retriever ready (vector k=%d, bm25 k=%d, weights=%s).",
        RETRIEVER_K,
        RETRIEVER_K,
        ENSEMBLE_WEIGHTS,
    )
    return HybridRetriever(
        vector_store, bm25_retriever, ENSEMBLE_WEIGHTS, RETRIEVER_K, excluded_acts
    )
