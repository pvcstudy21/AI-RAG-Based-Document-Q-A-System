from __future__ import annotations

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from rag_app.config import Settings


class QdrantStore:
    """Qdrant adapter using persistent embedded mode or a remote Qdrant endpoint."""

    def __init__(self, settings: Settings, embeddings):
        self.settings = settings
        self.embeddings = embeddings
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.qdrant_path))

        self.collection_name = settings.qdrant_collection
        if not self.client.collection_exists(self.collection_name):
            dimension = len(embeddings.embed_query("dimension probe"))
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embeddings,
            distance=models.Distance.COSINE,
        )

    def add_documents(self, chunks: list[Document]) -> None:
        if not chunks:
            return
        self.vector_store.add_documents(
            documents=chunks,
            ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks],
        )

    def delete_source(self, source_id: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source_id",
                        match=models.MatchValue(value=source_id),
                    )
                ]
            )
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=selector,
            wait=True,
        )

    def dense_search(
        self,
        query: str,
        k: int,
        source_ids: set[str] | None = None,
    ) -> list[tuple[Document, float]]:
        qdrant_filter = None
        if source_ids:
            qdrant_filter = models.Filter(
                should=[
                    models.FieldCondition(
                        key="metadata.source_id",
                        match=models.MatchValue(value=source_id),
                    )
                    for source_id in sorted(source_ids)
                ],
            )
        return self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=qdrant_filter,
        )

    def close(self) -> None:
        self.client.close()
