"""Legal-aware text chunking.

Uses a ``RecursiveCharacterTextSplitter`` with a separator hierarchy tuned for
legal documents, so splits prefer section/article/chapter boundaries before
falling back to paragraphs, sentences, and words.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE

LEGAL_SEPARATORS = [
    "\n## ",          # Markdown-style headers (from processed data)
    "\n### ",
    "\nSection ",     # Legal section boundaries
    "\nArticle ",     # Constitutional article boundaries
    "\nChapter ",     # Chapter boundaries
    "\n\n",           # Paragraph breaks
    "\n",             # Line breaks
    ". ",             # Sentence boundaries
    " ",              # Word boundaries (last resort)
]


def get_legal_chunker(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Build a legal-aware recursive character splitter.

    Args:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        A configured ``RecursiveCharacterTextSplitter``.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=LEGAL_SEPARATORS,
        length_function=len,
    )
