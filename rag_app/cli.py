from __future__ import annotations

import argparse
import json
import sys
from rag_app.catalog import ChunkCatalog
from rag_app.config import PROJECT_ROOT, Settings
from rag_app.evaluation import run_evaluation
from rag_app.service import RAGEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-rag",
        description="Local document ingestion, hybrid retrieval, and grounded Q&A.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse, chunk, embed, and index files")
    ingest.add_argument("paths", nargs="+", help="PDF, DOCX, TXT, Markdown, or HTML files")

    ask = subparsers.add_parser("ask", help="Ask a question against indexed documents")
    ask.add_argument("question")
    ask.add_argument("--source", help="Limit retrieval to an indexed filename or source ID")
    ask.add_argument("--top-k", type=int, default=None)

    retrieve = subparsers.add_parser("retrieve", help="Inspect ranked chunks without generation")
    retrieve.add_argument("question")
    retrieve.add_argument("--source", help="Limit retrieval to an indexed filename or source ID")
    retrieve.add_argument("--top-k", type=int, default=None)

    summarize = subparsers.add_parser("summarize", help="Create a map-reduce document summary")
    summarize.add_argument("--source", required=True, help="Indexed filename or source ID")

    subparsers.add_parser("list", help="List indexed source files")
    subparsers.add_parser("warmup", help="Download/load embedding and reranker models")

    evaluate = subparsers.add_parser("evaluate", help="Run retrieval and generation evaluation")
    evaluate.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "evals/golden_qa.json"),
    )
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--retrieval-only", action="store_true")
    evaluate.add_argument("--output", default=None)

    remove = subparsers.add_parser("remove", help="Remove a source file from the index")
    remove.add_argument("source", help="Indexed filename or source ID")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = Settings()
    engine = None
    try:
        if args.command == "list":
            settings.ensure_directories()
            sources = ChunkCatalog(settings).list_sources()
            if not sources:
                print("No documents are indexed yet.")
            else:
                print(f"{'SOURCE':40} {'CHUNKS':>8} {'SOURCE ID'}")
                for item in sources:
                    print(
                        f"{item['source_name'][:40]:40} {item['chunk_count']:>8} "
                        f"{item['source_id']}"
                    )
            return

        engine = RAGEngine(settings)
        if args.command == "ingest":
            results = engine.ingest_many(args.paths)
            for result in results:
                print(
                    f"Indexed {result['source_name']}: {result['parsed_documents']} parsed "
                    f"document(s) -> {result['chunks']} chunks (source_id={result['source_id']})"
                )
        elif args.command == "retrieve":
            hits = engine.retrieve(args.question, source=args.source, top_k=args.top_k)
            if not hits:
                print("No sufficiently relevant chunks found.")
            for index, item in enumerate(hits, start=1):
                metadata = item.document.metadata
                print(f"\n[{index}] {metadata.get('source_name')} | "
                      f"page={metadata.get('page_number')} | section={metadata.get('section_title')}")
                print(
                    f"dense_cosine={item.dense_similarity} | bm25_rank={item.bm25_rank} | "
                    f"rrf={item.rrf_score:.5f} | rerank={item.rerank_score}"
                )
                print(item.document.page_content[:1200])
        elif args.command == "ask":
            result = engine.ask(args.question, source=args.source, top_k=args.top_k)
            print("\n" + result.answer)
            if result.citations:
                print("\nRetrieved evidence:")
                for index, citation in enumerate(result.citations, start=1):
                    print(f"  [S{index}] {citation.citation}")
        elif args.command == "summarize":
            print(engine.summarize(args.source))
        elif args.command == "warmup":
            print("Embedding model loaded.")
            if settings.reranker_enabled:
                engine.retriever._get_reranker()
                print("Cross-encoder reranker loaded.")
            if settings.llm_provider == "ollama":
                print(f"Generation provider: Ollama ({settings.ollama_model})")
                print("Check it with: ollama list")
            else:
                _ = engine.generator
                print("Transformers generation model loaded.")
        elif args.command == "evaluate":
            summary = run_evaluation(
                engine,
                args.dataset,
                k=args.k,
                retrieval_only=args.retrieval_only,
                output_path=args.output,
            )
            print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
            print(f"\nPer-question details: {summary['report_path']}")
        elif args.command == "remove":
            source_ids = engine.resolve_source_ids(args.source)
            removed = 0
            for source_id in source_ids or set():
                removed += engine.remove(source_id)
            print(f"Removed {removed} catalog chunk(s) from the selected source(s).")
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()
