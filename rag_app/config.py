from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
# Set the Hugging Face cache location before any HF package is imported.
_hf_home_raw = Path(os.getenv("HF_HOME", "storage/huggingface")).expanduser()
_hf_home = _hf_home_raw if _hf_home_raw.is_absolute() else (PROJECT_ROOT / _hf_home_raw).resolve()
os.environ.setdefault("HF_HOME", str(_hf_home))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _path_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    qdrant_path: Path = _path_env("QDRANT_PATH", "storage/qdrant")
    qdrant_url: str = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "").strip()
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "document_chunks")
    catalog_path: Path = _path_env("CHUNK_CATALOG_PATH", "storage/chunks.jsonl")
    upload_dir: Path = _path_env("UPLOAD_DIR", "data/uploads")
    hf_home: Path = _path_env("HF_HOME", "storage/huggingface")

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "20"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "20"))
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))
    reranker_enabled: bool = _bool_env("RERANKER_ENABLED", True)
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    min_rerank_score: float = float(os.getenv("MIN_RERANK_SCORE", "0.10"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    transformers_generation_model: str = os.getenv(
        "TRANSFORMERS_GENERATION_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
    )
    generation_max_new_tokens: int = int(os.getenv("GENERATION_MAX_NEW_TOKENS", "450"))
    generation_temperature: float = float(os.getenv("GENERATION_TEMPERATURE", "0.1"))

    def ensure_directories(self) -> None:
        self.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
