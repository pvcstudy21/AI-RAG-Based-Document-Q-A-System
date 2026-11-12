from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from rag_app.config import Settings
from rag_app.service import RAGEngine

st.set_page_config(page_title="Document RAG Studio", page_icon="📚", layout="wide")
st.title("📚 Document RAG Studio")
st.caption("Upload a document, ask grounded questions, inspect citations, and evaluate retrieval.")


@st.cache_resource(show_spinner="Loading embedding model and Qdrant index...")
def get_engine() -> RAGEngine:
    return RAGEngine(Settings())


try:
    engine = get_engine()
except Exception as exc:
    st.error(f"Could not initialize the RAG engine: {exc}")
    st.info("Check the setup instructions and your model/cache configuration in README.md.")
    st.stop()

with st.sidebar:
    st.header("1. Add documents")
    uploads = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md", "markdown", "html", "htm"],
        accept_multiple_files=True,
    )
    if st.button("Ingest uploaded files", type="primary", disabled=not uploads):
        for uploaded in uploads or []:
            destination = engine.settings.upload_dir / Path(uploaded.name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = uploaded.getvalue()
            digest = hashlib.sha256(content).hexdigest()
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == digest:
                st.info(f"{uploaded.name} is already on disk; refreshing its index.")
            destination.write_bytes(content)
            with st.spinner(f"Parsing, chunking, embedding, and indexing {uploaded.name}..."):
                result = engine.ingest(destination)
            st.success(f"{result['source_name']}: indexed {result['chunks']} chunks.")
        st.rerun()

    st.divider()
    st.header("2. Choose the knowledge scope")
    sources = engine.list_sources()
    source_names = [item["source_name"] for item in sources]
    selected_label = st.selectbox("Search in", ["All indexed documents", *source_names])
    selected_source = None if selected_label == "All indexed documents" else selected_label
    if sources:
        with st.expander("Indexed documents"):
            for source in sources:
                st.write(f"**{source['source_name']}** — {source['chunk_count']} chunks")

    st.divider()
    st.header("3. Summarize")
    if source_names:
        summary_source = st.selectbox("Document to summarize", source_names)
        if st.button("Generate document summary"):
            with st.spinner("Running map-reduce summarization..."):
                try:
                    st.session_state["latest_summary"] = engine.summarize(summary_source)
                except Exception as exc:
                    st.error(f"Summary failed: {exc}")
        if st.session_state.get("latest_summary"):
            st.markdown(st.session_state["latest_summary"])

st.subheader("Ask your documents")
st.caption("Select one document to keep the answer scoped to that book/file, or search all indexed documents.")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Evidence and retrieval diagnostics"):
                for citation in message["citations"]:
                    st.markdown(f"**{citation['citation']}**")
                    st.caption(
                        f"cosine={citation['dense_similarity']} · "
                        f"BM25 rank={citation['bm25_rank']} · "
                        f"RRF={citation['rrf_score']:.5f} · "
                        f"reranker={citation['rerank_score']}"
                    )
                    st.write(citation["page_content"][:900])

question = st.chat_input("Ask a question about the selected document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            try:
                result = engine.ask(question, source=selected_source)
                st.markdown(result.answer)
                citation_rows = [
                    {
                        "citation": item.citation,
                        "dense_similarity": item.dense_similarity,
                        "bm25_rank": item.bm25_rank,
                        "rrf_score": item.rrf_score,
                        "rerank_score": item.rerank_score,
                        "page_content": item.page_content,
                    }
                    for item in result.citations
                ]
                if citation_rows:
                    with st.expander("Evidence and retrieval diagnostics"):
                        for index, citation in enumerate(citation_rows, start=1):
                            st.markdown(f"**[S{index}] {citation['citation']}**")
                            st.caption(
                                f"cosine={citation['dense_similarity']} · "
                                f"BM25 rank={citation['bm25_rank']} · "
                                f"RRF={citation['rrf_score']:.5f} · "
                                f"reranker={citation['rerank_score']}"
                            )
                            st.write(citation["page_content"][:900])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "citations": citation_rows,
                    }
                )
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")

st.divider()
st.caption(
    "Pipeline: parser → RecursiveCharacterTextSplitter → Hugging Face embeddings → "
    "Qdrant cosine search + BM25 → Reciprocal Rank Fusion → cross-encoder reranker → "
    "citation-grounded generation."
)
