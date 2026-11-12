from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import fitz
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm"}


def make_source_id(path: Path) -> str:
    """Stable for a path, so re-ingesting the same file replaces its old chunks."""
    normalized = str(path.expanduser().resolve()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _metadata(path: Path, source_id: str, **extra: object) -> dict:
    return {
        "source_id": source_id,
        "source_name": path.name,
        "source_path": str(path.resolve()),
        "file_type": path.suffix.lower().lstrip("."),
        **extra,
    }


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Keep Markdown heading context attached to its section before chunking."""
    matches = list(re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", text))
    if not matches:
        return [("", text)] if text.strip() else []
    sections: list[tuple[str, str]] = []
    if text[: matches[0].start()].strip():
        sections.append(("Introduction", text[: matches[0].start()].strip()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        body = text[match.end() : end].strip()
        if body:
            sections.append((title, f"{title}\n\n{body}"))
    return sections


def load_file(path: str | Path) -> list[Document]:
    """Parse one supported file into LangChain Documents with source metadata.

    PDF pages are separate Documents to preserve page citations. Markdown headings
    are separate Documents to preserve useful section boundaries. DOCX page numbers
    are not reliable in python-docx, so heading names are retained instead.
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    source_id = make_source_id(file_path)
    documents: list[Document] = []

    if suffix == ".pdf":
        with fitz.open(file_path) as pdf:
            for page_index, page in enumerate(pdf):
                text = page.get_text("text", sort=True).strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata=_metadata(file_path, source_id, page_number=page_index + 1),
                        )
                    )
        if not documents:
            raise ValueError(
                f"No extractable text found in {file_path.name}. It may be scanned/image-only; "
                "OCR is not enabled in this starter project."
            )
        return documents

    if suffix == ".docx":
        docx = DocxDocument(file_path)
        current_heading = ""
        section_parts: list[str] = []
        for paragraph in docx.paragraphs:
            content = paragraph.text.strip()
            if not content:
                continue
            style_name = (paragraph.style.name if paragraph.style else "").lower()
            if style_name.startswith("heading"):
                if section_parts:
                    documents.append(
                        Document(
                            page_content="\n\n".join(section_parts),
                            metadata=_metadata(file_path, source_id, section_title=current_heading),
                        )
                    )
                current_heading = content
                section_parts = [content]
            else:
                section_parts.append(content)
        for table in docx.tables:
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells]
                if any(values):
                    section_parts.append(" | ".join(values))
        if section_parts:
            documents.append(
                Document(
                    page_content="\n\n".join(section_parts),
                    metadata=_metadata(file_path, source_id, section_title=current_heading),
                )
            )
        return documents

    raw = file_path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(raw, "html.parser")
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        raw = soup.get_text("\n", strip=True)
        if title:
            raw = f"{title}\n\n{raw}"

    if suffix in {".md", ".markdown"}:
        for heading, body in _markdown_sections(raw):
            documents.append(
                Document(
                    page_content=body,
                    metadata=_metadata(file_path, source_id, section_title=heading),
                )
            )
    else:
        if raw.strip():
            documents.append(
                Document(page_content=raw, metadata=_metadata(file_path, source_id))
            )

    if not documents:
        raise ValueError(f"No readable text was found in {file_path.name}.")
    return documents


def load_files(paths: Iterable[str | Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        documents.extend(load_file(path))
    return documents
