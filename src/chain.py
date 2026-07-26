"""The conversational RAG chain for Legal Agent v2.

Pipeline per turn: contextualize the query against chat history, classify it into
an optional metadata filter, retrieve via the hybrid (vector + BM25) retriever,
rerank the candidates with a cheap LLM, then stream the grounded answer.

A manual chain is used (rather than ``create_retrieval_chain``) so that the
final answer can be cleanly streamed token-by-token to the Streamlit UI.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    CHAT_TEMPERATURE,
    GEMINI_CHAT_MODEL,
    GEMINI_LITE_MODEL,
    RERANKER_TOP_K,
)
from src.classifier import classify_query
from src.prompts import CONTEXTUALIZE_PROMPT, QA_SYSTEM_PROMPT
from src.reranker import rerank_documents
from src.retriever import build_hybrid_retriever
from src.utils import load_bm25_documents, trim_chat_history

logger = logging.getLogger(__name__)


class LegalRAGChain:
    """Conversational RAG chain with streaming generation.

    Exposes a ``.stream()`` method that yields answer tokens and records the
    source documents used for the latest turn on ``self.last_sources``.
    """

    def __init__(self, vector_store: Any, api_key: str) -> None:
        self._chat_llm = ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            temperature=CHAT_TEMPERATURE,
            google_api_key=api_key,
        )
        # Cheap model reserved for the classifier/reranker (Phase 3).
        self._lite_llm = ChatGoogleGenerativeAI(
            model=GEMINI_LITE_MODEL,
            temperature=0.0,
            google_api_key=api_key,
        )

        bm25_docs = load_bm25_documents(vector_store)
        self._retriever = build_hybrid_retriever(vector_store, bm25_docs)

        self._contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self._qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Populated after each call to ``stream``.
        self.last_sources: list[Document] = []
        self.last_filter: dict = {}

    # -- internal steps -----------------------------------------------------

    def _contextualize(self, query: str, history: list[Any]) -> str:
        """Rewrite a follow-up question into a standalone query.

        Falls back to the raw query when there is no history or on error.
        """
        if not history:
            return query
        try:
            chain = self._contextualize_prompt | self._chat_llm
            result = chain.invoke({"input": query, "chat_history": history})
            standalone = getattr(result, "content", "") or query
            logger.debug("Contextualized query: %s", standalone)
            return standalone.strip()
        except Exception:
            logger.exception("Contextualization failed; using raw query.")
            return query

    def _retrieve(self, query: str) -> list[Document]:
        """Classify, retrieve (optionally filtered), then rerank.

        1. The cheap LLM classifies the query into a ChromaDB ``where`` filter
           (empty when the query is vague / not confidently one category).
        2. The hybrid retriever fetches ``RETRIEVER_K`` candidates — vector search
           is narrowed by the filter, BM25 always searches everything.
        3. The cheap LLM reranks and we keep the top ``RERANKER_TOP_K``.
        """
        try:
            where = classify_query(query, self._lite_llm)
        except Exception:
            logger.exception("Classification failed; proceeding without a filter.")
            where = {}
        self.last_filter = where

        try:
            docs = self._retriever.invoke(query, where=where or None)
        except Exception:
            logger.exception("Retrieval failed; returning no documents.")
            return []

        return rerank_documents(query, docs, self._lite_llm, top_k=RERANKER_TOP_K)

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        """Render retrieved documents into the prompt context block."""
        blocks: list[str] = []
        for i, doc in enumerate(documents, start=1):
            meta = doc.metadata or {}
            header = " | ".join(
                part
                for part in (
                    meta.get("act_name"),
                    (f"Section {meta['section_number']}" if meta.get("section_number") else None),
                    meta.get("title"),
                )
                if part
            )
            header = header or f"Document {i}"
            blocks.append(f"[{i}] {header}\n{doc.page_content}")
        return "\n\n".join(blocks) if blocks else "No relevant documents found."

    # -- public API ---------------------------------------------------------

    def stream(self, inputs: dict) -> Iterator[str]:
        """Stream the answer for a user turn, yielding tokens.

        Args:
            inputs: A dict with keys ``input`` (the user question) and
                ``chat_history`` (list of LangChain messages).

        Yields:
            Answer text chunks as they are generated.
        """
        query: str = inputs["input"]
        history: list[Any] = trim_chat_history(inputs.get("chat_history", []))

        standalone = self._contextualize(query, history)
        documents = self._retrieve(standalone)
        self.last_sources = documents

        context = self._format_context(documents)
        messages = self._qa_prompt.format_messages(
            input=query,
            chat_history=history,
            context=context,
        )

        try:
            for chunk in self._chat_llm.stream(messages):
                text = getattr(chunk, "content", "")
                if text:
                    yield text
        except Exception as exc:
            logger.exception("Generation streaming failed.")
            yield f"\n\nAn error occurred while generating the response: {exc}"


def build_rag_chain(vector_store: Any, api_key: str) -> LegalRAGChain:
    """Build the full conversational RAG chain.

    The returned object exposes a ``.stream(inputs)`` method that yields answer
    tokens and records source documents on ``.last_sources``.

    Args:
        vector_store: A LangChain Chroma vector store.
        api_key: The Google Generative AI API key.

    Returns:
        A configured :class:`LegalRAGChain`.
    """
    logger.info("Building RAG chain with chat model '%s'.", GEMINI_CHAT_MODEL)
    return LegalRAGChain(vector_store, api_key)
