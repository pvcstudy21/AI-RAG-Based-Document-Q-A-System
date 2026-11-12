from __future__ import annotations

import hashlib
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.config import Settings


def build_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """A strong general-purpose default for prose, books, and technical docs.

    It prefers paragraph/line/sentence boundaries, then falls back to spaces and
    characters. Page and section boundaries are preserved by the parser because
    each parsed page/section is split independently.
    """
    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        strip_whitespace=True,
    )


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = build_splitter(settings)
    chunks = splitter.split_documents(documents)
    for chunk_index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        source_id = str(metadata["source_id"])
        digest = hashlib.sha256(
            f"{source_id}:{chunk_index}:{chunk.page_content}".encode("utf-8")
        ).hexdigest()[:32]
        # Qdrant point IDs accept UUIDs or unsigned integers; format the stable hash as UUID.
        chunk_id = str(uuid.UUID(digest))
        metadata.update({"chunk_index": chunk_index, "chunk_id": chunk_id})
        chunk.metadata = metadata
    return chunks
