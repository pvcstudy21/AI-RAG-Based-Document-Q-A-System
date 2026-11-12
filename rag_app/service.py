from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rag_app.answering import AnswerResult, answer_question, summarize_chunks
from rag_app.catalog import ChunkCatalog
from rag_app.chunking import split_documents
from rag_app.config import Settings
from rag_app.embeddings import build_embeddings
from rag_app.generation import build_generator
from rag_app.loaders import load_file, make_source_id
from rag_app.retrieval import HybridRetriever
from rag_app.vectorstore import QdrantStore


class RAGEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.catalog = ChunkCatalog(self.settings)
        self.embeddings = build_embeddings(self.settings)
        self.store = QdrantStore(self.settings, self.embeddings)
        self.retriever = HybridRetriever(self.settings, self.store, self.catalog)
        self._generator = None

    @property
    def generator(self):
        if self._generator is None:
            self._generator = build_generator(self.settings)
        return self._generator

    def ingest(self, path: str | Path) -> dict:
        file_path = Path(path).expanduser().resolve()
        parsed = load_file(file_path)
        chunks = split_documents(parsed, self.settings)
        source_id = make_source_id(file_path)
        # Replace old Qdrant points and catalog rows for this exact file path.
        self.store.delete_source(source_id)
        self.store.add_documents(chunks)
        self.catalog.replace_source(source_id, chunks)
        return {
            "source_id": source_id,
            "source_name": file_path.name,
            "parsed_documents": len(parsed),
            "chunks": len(chunks),
        }

    def ingest_many(self, paths: list[str | Path]) -> list[dict]:
        return [self.ingest(path) for path in paths]

    def remove(self, source_id: str) -> int:
        self.store.delete_source(source_id)
        return self.catalog.remove_source(source_id)

    def list_sources(self) -> list[dict]:
        return self.catalog.list_sources()

    def resolve_source_ids(self, source_name_or_id: str | None) -> set[str] | None:
        if not source_name_or_id:
            return None
        needle = source_name_or_id.strip().casefold()
        matches = {
            item["source_id"]
            for item in self.list_sources()
            if item["source_id"].casefold() == needle
            or item["source_name"].casefold() == needle
        }
        if not matches:
            known = ", ".join(item["source_name"] for item in self.list_sources()) or "(none)"
            raise ValueError(f"No indexed source matches {source_name_or_id!r}. Indexed: {known}")
        return matches

    def ask(
        self,
        question: str,
        *,
        source: str | None = None,
        top_k: int | None = None,
    ) -> AnswerResult:
        source_ids = self.resolve_source_ids(source)
        retrieved = self.retriever.retrieve(
            question,
            top_k=top_k,
            source_ids=source_ids,
        )
        return answer_question(question, retrieved, self.generator, self.settings)

    def retrieve(self, question: str, *, source: str | None = None, top_k: int | None = None):
        source_ids = self.resolve_source_ids(source)
        return self.retriever.retrieve(question, top_k=top_k, source_ids=source_ids)

    def summarize(self, source: str) -> str:
        source_ids = self.resolve_source_ids(source)
        chunks = self.catalog.documents(source_ids)
        chunks.sort(
            key=lambda doc: (
                doc.metadata.get("page_number", 0),
                doc.metadata.get("chunk_index", 0),
            )
        )
        return summarize_chunks(chunks, self.generator, self.settings)

    def close(self) -> None:
        self.store.close()
