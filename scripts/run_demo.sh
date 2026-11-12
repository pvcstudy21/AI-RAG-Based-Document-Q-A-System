#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after activating .venv.
python -m rag_app.cli ingest examples/sample_book.md
python -m rag_app.cli list
python -m rag_app.cli retrieve \
  "How does BM25 complement semantic retrieval?" \
  --source sample_book.md
python -m rag_app.cli ask \
  "How does BM25 complement semantic retrieval?" \
  --source sample_book.md
python -m rag_app.cli ask \
  "What is the exact ISBN of this sample book?" \
  --source sample_book.md
python -m rag_app.cli evaluate --dataset evals/golden_qa.json --k 5 --retrieval-only
