from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from rag_app.config import Settings


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    settings.ensure_directories()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.embedding_batch_size,
        },
        query_encode_kwargs={"normalize_embeddings": True},
        show_progress=True,
    )
