# --- SQLite shim for Streamlit Cloud (MUST be first, before any other import) ---
# Streamlit Cloud ships an old system sqlite3 that ChromaDB rejects. Swap in the
# modern pysqlite3 build. Wrapped so local dev (where pysqlite3 may be absent)
# still works.
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import logging
import os
import traceback

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_DIM,
    GEMINI_EMBEDDING_MODEL,
    GOOGLE_API_KEY,
    LEGAL_CATEGORIES,
)
from src.chain import build_rag_chain
from src.utils import format_sources

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("legal_agent")

EXAMPLE_QUESTIONS = [
    "What is the punishment for murder under the Bharatiya Nyaya Sanhita?",
    "What are my fundamental rights under the Indian Constitution?",
    "How do I file a consumer complaint?",
    "What is the penalty for driving without a licence?",
]

INITIAL_GREETING = (
    "Hello! I am your Legal Agent for Indian law. Ask me a question and I'll "
    "answer using grounded, cited sources from my knowledge base."
)


# --- Resource loading (cached) --------------------------------------------
@st.cache_resource
def load_vector_store() -> Chroma:
    """Load the embedding function and Chroma vector store (cached)."""
    logger.info("Loading vector store from %s ...", DB_PATH)
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found. Set it in .env or Streamlit secrets.")
        st.stop()
    if not os.path.exists(DB_PATH):
        st.error(f"ChromaDB not found at {DB_PATH}. Build the database first.")
        st.stop()
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_query",
            output_dimensionality=EMBEDDING_DIM,
        )
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        count = vector_store._collection.count()
        logger.info("Vector store loaded: collection '%s' has %d chunks.", COLLECTION_NAME, count)
        if count == 0:
            st.warning(f"Collection '{COLLECTION_NAME}' is empty.")
        return vector_store
    except Exception as exc:
        st.error(f"Failed to initialize vector store: {exc}")
        traceback.print_exc()
        st.stop()


@st.cache_resource
def load_chain(_vector_store: Chroma):
    """Build the conversational RAG chain (cached)."""
    logger.info("Building RAG chain ...")
    try:
        return build_rag_chain(_vector_store, GOOGLE_API_KEY)
    except Exception as exc:
        st.error(f"Failed to build RAG chain: {exc}")
        traceback.print_exc()
        st.stop()


# --- Page setup ------------------------------------------------------------
st.set_page_config(page_title="Legal Agent — Indian Law", page_icon="⚖️", layout="centered")


def reset_chat() -> None:
    """Clear the conversation and restore the greeting."""
    st.session_state.chat_history = [AIMessage(content=INITIAL_GREETING)]


# --- Session state ---------------------------------------------------------
if "chat_history" not in st.session_state:
    reset_chat()
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

vector_store = load_vector_store()
rag_chain = load_chain(vector_store)


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ Legal Agent")
    st.caption("Conversational RAG for Indian law")
    st.button("🆕 New Chat", use_container_width=True, on_click=reset_chat)

    st.divider()
    st.selectbox(
        "Filter by category",
        options=["All categories", *LEGAL_CATEGORIES],
        key="category_filter",
        help="Category-scoped retrieval arrives in a later phase.",
    )

    st.divider()
    st.subheader("Example questions")
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if st.button(question, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_query = question

    st.divider()
    st.caption(
        "⚖️ This tool provides general legal information, not legal advice. "
        "Consult a qualified lawyer for your specific situation."
    )


# --- Main chat area --------------------------------------------------------
st.title("⚖️ Your Personal Legal Agent")
st.info("Ask a question about Indian law. Answers are grounded in cited sources.")

for message in st.session_state.chat_history:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)


def render_sources(documents) -> None:
    """Render retrieved sources in an expandable section."""
    if not documents:
        return
    with st.expander(f"📚 Sources ({len(documents)})"):
        for i, src in enumerate(format_sources(documents), start=1):
            heading = src["title"] or f"Source {i}"
            if src["act_name"]:
                heading = f"{src['act_name']} — {heading}"
            st.markdown(f"**{i}. {heading}**")
            meta_bits = [b for b in (src["category"], src["chunk_type"]) if b]
            if meta_bits:
                st.caption(" · ".join(meta_bits))
            st.write(src["snippet"])
            if src["source"]:
                st.caption(f"Source: {src['source']}")
            st.divider()


def handle_query(user_query: str) -> None:
    """Append the user turn, stream the answer, and record sources."""
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                rag_chain.stream(
                    {
                        "input": user_query,
                        "chat_history": st.session_state.chat_history[:-1],
                    }
                )
            )
            render_sources(rag_chain.last_sources)
            st.session_state.chat_history.append(AIMessage(content=answer))
        except Exception as exc:
            logger.exception("Error while answering query.")
            st.error(f"An error occurred: {exc}")
            st.session_state.chat_history.append(
                AIMessage(content="Sorry, I encountered an error. Please try again.")
            )


# Resolve input: a clicked example question, or the chat box.
typed_query = st.chat_input("Type your legal question here…")
query = typed_query
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query and query.strip():
    handle_query(query.strip())
