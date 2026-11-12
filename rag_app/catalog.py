from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain_core.documents import Document

from rag_app.config import Settings


class ChunkCatalog:
    """Small durable text/metadata catalog used to rebuild BM25 on process start."""

    def __init__(self, settings: Settings):
        self.path = settings.catalog_path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read_records(self) -> list[dict]:
        if not self.path.exists():
            return []
        records: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in chunk catalog at line {line_number}: {exc}"
                    ) from exc
                if "page_content" in record and "metadata" in record:
                    records.append(record)
        return records

    def documents(self, source_ids: set[str] | None = None) -> list[Document]:
        records = self.read_records()
        documents = []
        for record in records:
            metadata = record["metadata"]
            if source_ids is not None and metadata.get("source_id") not in source_ids:
                continue
            documents.append(
                Document(page_content=record["page_content"], metadata=metadata)
            )
        return documents

    def replace_source(self, source_id: str, chunks: list[Document]) -> None:
        retained = [
            row for row in self.read_records()
            if row.get("metadata", {}).get("source_id") != source_id
        ]
        retained.extend(
            {"page_content": chunk.page_content, "metadata": chunk.metadata}
            for chunk in chunks
        )
        self._write_records(retained)

    def remove_source(self, source_id: str) -> int:
        old = self.read_records()
        retained = [
            row for row in old
            if row.get("metadata", {}).get("source_id") != source_id
        ]
        self._write_records(retained)
        return len(old) - len(retained)

    def _write_records(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(self.path)

    def list_sources(self) -> list[dict]:
        grouped: dict[str, dict] = {}
        for record in self.read_records():
            metadata = record["metadata"]
            source_id = metadata.get("source_id")
            if not source_id:
                continue
            if source_id not in grouped:
                grouped[source_id] = {
                    "source_id": source_id,
                    "source_name": metadata.get("source_name", "Unknown"),
                    "source_path": metadata.get("source_path", ""),
                    "chunk_count": 0,
                }
            grouped[source_id]["chunk_count"] += 1
        return sorted(grouped.values(), key=lambda item: item["source_name"].lower())
