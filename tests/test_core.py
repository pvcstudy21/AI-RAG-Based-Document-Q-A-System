from pathlib import Path

from langchain_core.documents import Document

from rag_app.answering import SourceCitation, validate_citations
from rag_app.chunking import split_documents
from rag_app.config import Settings
from rag_app.evaluation import retrieval_metrics
from rag_app.loaders import load_file
from rag_app.retrieval import tokenize


def test_markdown_parser_preserves_sections(tmp_path: Path):
    path = tmp_path / "tiny_book.md"
    path.write_text("# First\n\nAlpha text.\n\n# Second\n\nBeta text.", encoding="utf-8")
    documents = load_file(path)
    assert [doc.metadata["section_title"] for doc in documents] == ["First", "Second"]
    assert "Alpha text" in documents[0].page_content


def test_chunking_adds_stable_ids_and_keeps_metadata(tmp_path: Path):
    settings = Settings(
        chunk_size=80,
        chunk_overlap=10,
        catalog_path=tmp_path / "chunks.jsonl",
        qdrant_path=tmp_path / "qdrant",
        upload_dir=tmp_path / "uploads",
        hf_home=tmp_path / "hf",
    )
    text = "Paragraph one. " * 20 + "\n\n" + "Paragraph two. " * 20
    docs = [Document(page_content=text, metadata={"source_id": "abc", "source_name": "x.txt"})]
    chunks = split_documents(docs, settings)
    assert len(chunks) > 1
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)
    assert all(chunk.metadata["source_id"] == "abc" for chunk in chunks)


def test_tokenize_handles_case_and_punctuation():
    assert tokenize("BM25 + Dense Search, RAG-based!") == ["bm25", "dense", "search", "rag-based"]


def test_retrieval_metrics():
    metrics = retrieval_metrics(["wrong", "right", "also-right"], {"right", "also-right"}, 3)
    assert metrics["precision@3"] == 2 / 3
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg@3"] <= 1


def test_retrieval_metrics_empty_ground_truth():
    metrics = retrieval_metrics(["x"], set(), 5)
    assert metrics["precision@5"] is None


def test_citation_validator_rejects_unknown_markers_and_adds_sources():
    citations = [
        SourceCitation("a", "book.pdf", "book.pdf, p. 2", "id1", "text", 0.8, 1, 0.03, 0.9)
    ]
    answer, has_citation = validate_citations("A claim [S99].", citations)
    assert "[S99]" not in answer
    assert "[S1]" in answer
    assert has_citation
