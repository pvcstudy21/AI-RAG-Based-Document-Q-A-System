.PHONY: install test ingest-demo evaluate ui

install:
	python -m pip install -e ".[dev]"

test:
	pytest

ingest-demo:
	python -m rag_app.cli ingest examples/sample_book.md

evaluate:
	python -m rag_app.cli evaluate --dataset evals/golden_qa.json --k 5

ui:
	streamlit run app.py
