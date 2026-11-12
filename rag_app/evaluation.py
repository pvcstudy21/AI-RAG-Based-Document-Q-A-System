from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from rag_app.answering import answer_question
from rag_app.service import RAGEngine


def _relevant_ids(documents, phrases: list[str]) -> set[str]:
    normalized = [phrase.casefold() for phrase in phrases]
    return {
        str(doc.metadata.get("chunk_id"))
        for doc in documents
        if any(phrase in doc.page_content.casefold() for phrase in normalized)
    }


def retrieval_metrics(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> dict:
    if not relevant_ids:
        return {
            f"precision@{k}": None,
            f"recall@{k}": None,
            "mrr": None,
            f"ndcg@{k}": None,
            "relevant_count": 0,
            "retrieved_relevant": 0,
        }
    top_ids = retrieved_ids[:k]
    hits = [1 if item in relevant_ids else 0 for item in top_ids]
    hit_count = sum(hits)
    first_rank = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
    return {
        f"precision@{k}": hit_count / k,
        f"recall@{k}": hit_count / len(relevant_ids),
        "mrr": 1 / first_rank if first_rank else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        "relevant_count": len(relevant_ids),
        "retrieved_relevant": hit_count,
    }


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def run_evaluation(
    engine: RAGEngine,
    dataset_path: str | Path,
    *,
    k: int = 5,
    retrieval_only: bool = False,
    output_path: str | Path | None = None,
) -> dict:
    dataset_file = Path(dataset_path)
    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    all_rows: list[dict] = []
    for case in dataset["cases"]:
        question = case["question"]
        source = case.get("source")
        source_ids = engine.resolve_source_ids(source) if source else None
        corpus = engine.catalog.documents(source_ids)
        relevant_ids = _relevant_ids(corpus, case.get("relevant_text_contains", []))
        retrieved = engine.retriever.retrieve(question, top_k=k, source_ids=source_ids)
        retrieved_ids = [str(item.document.metadata.get("chunk_id")) for item in retrieved]
        metrics = retrieval_metrics(retrieved_ids, relevant_ids, k)
        row = {
            "question": question,
            "source": source,
            "expected_abstention": bool(case.get("expected_abstention", False)),
            "retrieved": [
                {
                    "source_name": item.document.metadata.get("source_name"),
                    "page_number": item.document.metadata.get("page_number"),
                    "section_title": item.document.metadata.get("section_title"),
                    "chunk_id": item.document.metadata.get("chunk_id"),
                    "dense_similarity": item.dense_similarity,
                    "bm25_rank": item.bm25_rank,
                    "rrf_score": item.rrf_score,
                    "rerank_score": item.rerank_score,
                }
                for item in retrieved
            ],
            "retrieval_metrics": metrics,
        }
        if not retrieval_only:
            answer = answer_question(question, retrieved, engine.generator, engine.settings)
            expected_terms = case.get("expected_answer_terms", [])
            normalized_answer = _normalize_for_match(answer.answer)
            term_hits = sum(
                1 for term in expected_terms
                if _normalize_for_match(term) in normalized_answer
            )
            term_coverage = term_hits / len(expected_terms) if expected_terms else None
            valid_citation_ids = {
                f"S{index}" for index in range(1, len(answer.citations) + 1)
            }
            cited_ids = set(re.findall(r"\[(S\d+)\]", answer.answer))
            row["generation"] = {
                "answer": answer.answer,
                "abstained": answer.abstained,
                "expected_term_coverage": term_coverage,
                "citation_present": bool(cited_ids),
                "citation_ids_valid": cited_ids.issubset(valid_citation_ids),
                "citation_count": len(cited_ids),
            }
        all_rows.append(row)

    def mean_metric(key: str) -> float | None:
        values = [
            row["retrieval_metrics"][key]
            for row in all_rows
            if row["retrieval_metrics"].get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    summary = {
        "dataset": str(dataset_file),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "case_count": len(all_rows),
        "retrieval": {
            f"mean_precision@{k}": mean_metric(f"precision@{k}"),
            f"mean_recall@{k}": mean_metric(f"recall@{k}"),
            "mean_mrr": mean_metric("mrr"),
            f"mean_ndcg@{k}": mean_metric(f"ndcg@{k}"),
        },
        "cases": all_rows,
    }
    generation_rows = [row["generation"] for row in all_rows if "generation" in row]
    if generation_rows:
        term_values = [
            row["expected_term_coverage"]
            for row in generation_rows
            if row["expected_term_coverage"] is not None
        ]
        summary["generation"] = {
            "citation_coverage": sum(row["citation_present"] for row in generation_rows)
            / len(generation_rows),
            "valid_citation_rate": sum(row["citation_ids_valid"] for row in generation_rows)
            / len(generation_rows),
            "mean_expected_term_coverage": (
                sum(term_values) / len(term_values) if term_values else None
            ),
            "abstention_accuracy": sum(
                row["abstained"] == row["expected_abstention"]
                for row in generation_rows
            ) / len(generation_rows),
        }

    destination = Path(output_path) if output_path else engine.settings.project_root / "storage/evaluation/latest_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["report_path"] = str(destination.resolve())
    return summary
