from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag_app.catalog import ChunkCatalog
from rag_app.config import Settings
from rag_app.vectorstore import QdrantStore


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())


@dataclass
class RetrievedChunk:
    document: Document
    dense_similarity: float | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


class HybridRetriever:
    """Dense cosine + BM25 fused by RRF, optionally refined by a cross-encoder."""

    def __init__(self, settings: Settings, store: QdrantStore, catalog: ChunkCatalog):
        self.settings = settings
        self.store = store
        self.catalog = catalog
        self._reranker = None

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            from torch import nn

            self._reranker = CrossEncoder(
                self.settings.reranker_model,
                device="cpu",
                activation_fn=nn.Sigmoid(),
            )
        return self._reranker

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        final_k = top_k or self.settings.final_top_k
        corpus = self.catalog.documents(source_ids)
        if not corpus:
            return []

        dense_hits = self.store.dense_search(
            query,
            k=min(self.settings.dense_top_k, len(corpus)),
            source_ids=source_ids,
        )
        bm25 = BM25Retriever.from_documents(
            corpus,
            preprocess_func=tokenize,
            k=min(self.settings.bm25_top_k, len(corpus)),
        )
        lexical_hits = bm25.invoke(query)

        # Reciprocal Rank Fusion combines ranks, not incomparable raw score scales.
        fused: dict[str, RetrievedChunk] = {}
        for rank, (document, cosine_score) in enumerate(dense_hits, start=1):
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if not chunk_id:
                continue
            item = fused.setdefault(chunk_id, RetrievedChunk(document=document))
            item.dense_similarity = float(cosine_score)
            item.rrf_score += 1.0 / (self.settings.rrf_k + rank)

        for rank, document in enumerate(lexical_hits, start=1):
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if not chunk_id:
                continue
            item = fused.setdefault(chunk_id, RetrievedChunk(document=document))
            item.bm25_rank = rank
            item.rrf_score += 1.0 / (self.settings.rrf_k + rank)

        candidates = sorted(fused.values(), key=lambda item: item.rrf_score, reverse=True)
        candidates = candidates[: self.settings.rerank_candidates]
        if not candidates:
            return []

        if self.settings.reranker_enabled:
            reranker = self._get_reranker()
            pairs = [(query, item.document.page_content) for item in candidates]
            scores = reranker.predict(pairs, show_progress_bar=False, batch_size=16)
            for item, score in zip(candidates, scores):
                item.rerank_score = float(score)
            candidates.sort(
                key=lambda item: item.rerank_score if item.rerank_score is not None else -1,
                reverse=True,
            )
            # A weak top cross-encoder score is a conservative abstention signal.
            if candidates[0].rerank_score is not None:
                if candidates[0].rerank_score < self.settings.min_rerank_score:
                    return []
        else:
            candidates.sort(key=lambda item: item.rrf_score, reverse=True)

        return candidates[:final_k]
