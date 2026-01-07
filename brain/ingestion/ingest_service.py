"""
Princeps Brain Layer - Document & Code Ingestion Pipeline
=========================================================

Robust ingestion process that takes source data (Git repositories or documents)
and populates the Brain's database with content chunks, embeddings, and metadata.

Usage:
    from brain.ingestion import IngestService

    service = IngestService()
    result = service.ingest_document("/path/to/document.pdf")
    result = service.ingest_repository("/path/to/repo")
"""

import hashlib
import logging
import re
import subprocess
import traceback
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from ..core.db import (
    compute_content_hash,
    get_default_tenant_id,
    get_engine,
    get_or_create_operation,
    get_session,
    mark_operation_failed,
    mark_operation_started,
    mark_operation_success,
)

# Import from core module
from ..core.models import (
    DocChunk,
    Document,
    KnowledgeTypeEnum,
    OperationStatusEnum,
    OperationTypeEnum,
    Repository,
    Resource,
    ResourceDependency,
    ResourceTypeEnum,
    Tenant,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class IngestConfig:
    """Configuration for ingestion pipeline."""

    chunk_tokens: int = 800
    overlap_tokens: int = 100
    generate_embeddings: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    max_file_size_mb: float = 50.0
    parse_dependencies: bool = True
    scan_for_pii: bool = True
    scan_for_secrets: bool = True
    include_extensions: list[str] = field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".md",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".html",
            ".css",
            ".sql",
            ".pdf",
            ".rst",
            ".ipynb",
        ]
    )
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "__pycache__",
            "node_modules",
            ".git",
            ".venv",
            "venv",
            "dist",
            "build",
            ".egg-info",
            "*.pyc",
            "*.pyo",
        ]
    )
    continue_on_error: bool = True
    max_errors_per_batch: int = 10


@dataclass
class IngestResult:
    """Result of an ingestion operation."""

    success: bool
    operation_id: str | None = None
    documents_created: int = 0
    chunks_created: int = 0
    resources_created: int = 0
    dependencies_found: int = 0
    embeddings_generated: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# TEXT EXTRACTION
# =============================================================================


class TextExtractor:
    """Extract text content from various file formats."""

    _encoder = None

    @classmethod
    def get_encoder(cls):
        if cls._encoder is None:
            try:
                import tiktoken

                cls._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                cls._encoder = False
        return cls._encoder

    @staticmethod
    def count_tokens(text: str) -> int:
        encoder = TextExtractor.get_encoder()
        if encoder and encoder is not False:
            return len(encoder.encode(text))
        return max(1, len(text) // 4)

    @staticmethod
    def extract_pdf(path: str) -> tuple[str, int, dict[str, Any]]:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader

        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")

        full_text = "\n\n".join(texts)
        full_text = TextExtractor._clean_text(full_text)

        metadata = {}
        if reader.metadata:
            for key in ["title", "author", "subject", "creator"]:
                if hasattr(reader.metadata, key):
                    value = getattr(reader.metadata, key)
                    if value:
                        metadata[key] = str(value)

        return full_text, len(reader.pages), metadata

    @staticmethod
    def extract_text_file(path: str) -> tuple[str, dict[str, Any]]:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        content = TextExtractor._clean_text(content)
        return content, {"encoding": "utf-8", "line_count": content.count("\n") + 1}

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", "")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def extract(path: str) -> tuple[str, dict[str, Any]]:
        path = Path(path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            text, pages, metadata = TextExtractor.extract_pdf(str(path))
            metadata["page_count"] = pages
            return text, metadata
        elif ext == ".ipynb":
            import json

            with open(path, encoding="utf-8") as f:
                notebook = json.load(f)
            cells_text = []
            for cell in notebook.get("cells", []):
                source = "".join(cell.get("source", []))
                if cell.get("cell_type") == "code":
                    cells_text.append(f"```python\n{source}\n```")
                else:
                    cells_text.append(source)
            return "\n\n".join(cells_text), {"notebook": True, "cell_count": len(cells_text)}
        else:
            return TextExtractor.extract_text_file(str(path))


# =============================================================================
# TEXT CHUNKING
# =============================================================================


class TextChunker:
    """Split text into chunks for embedding."""

    def __init__(self, chunk_tokens: int = 800, overlap_tokens: int = 100):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self._encoder = TextExtractor.get_encoder()

    def chunk(self, text: str) -> list[dict[str, Any]]:
        if not text:
            return []

        if self._encoder and self._encoder is not False:
            return self._chunk_by_tokens(text)
        else:
            return self._chunk_by_chars(text)

    def _chunk_by_tokens(self, text: str) -> list[dict[str, Any]]:
        tokens = self._encoder.encode(text)
        chunks = []
        start_idx = 0
        chunk_index = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self._encoder.decode(chunk_tokens)

            start_char = len(self._encoder.decode(tokens[:start_idx])) if chunk_index > 0 else 0
            end_char = len(self._encoder.decode(tokens[:end_idx]))

            chunks.append(
                {
                    "content": chunk_text,
                    "index": chunk_index,
                    "start_char": start_char,
                    "end_char": end_char,
                    "token_count": len(chunk_tokens),
                    "char_count": len(chunk_text),
                }
            )

            # Break if we've reached the end of the tokens
            if end_idx >= len(tokens):
                break

            chunk_index += 1
            # Move forward, ensuring we always make progress
            next_start = end_idx - self.overlap_tokens
            # Ensure forward progress: must advance at least 1 token
            start_idx = max(start_idx + 1, next_start)

        return chunks

    def _chunk_by_chars(self, text: str) -> list[dict[str, Any]]:
        chunk_chars = self.chunk_tokens * 4
        overlap_chars = self.overlap_tokens * 4
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                for sep in [". ", ".\n", "\n\n", "! ", "? "]:
                    pos = text.rfind(sep, start + chunk_chars // 2, end)
                    if pos > 0:
                        end = pos + len(sep)
                        break

            chunk_text = text[start:end]
            chunks.append(
                {
                    "content": chunk_text,
                    "index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "token_count": TextExtractor.count_tokens(chunk_text),
                    "char_count": len(chunk_text),
                }
            )

            # Break if we've reached the end of the text
            if end >= len(text):
                break

            chunk_index += 1
            # Move forward, ensuring we always make progress
            next_start = end - overlap_chars
            # Ensure forward progress: must advance at least 1 char
            start = max(start + 1, next_start)

        return chunks


# =============================================================================
# EMBEDDING SERVICE
# =============================================================================


class EmbeddingService:
    """Generate embeddings for text chunks."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._available = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._available = True
            except ImportError:
                self._available = False
        return self._model

    @property
    def is_available(self) -> bool:
        if self._available is None:
            _ = self.model
        return self._available

    @property
    def dimension(self) -> int:
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384

    def embed(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> list[list[float]]:
        if not texts:
            return []

        if not self.is_available:
            return [self._mock_embedding(t) for t in texts]

        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True
        )
        return [emb.tolist() for emb in embeddings]

    def _mock_embedding(self, text: str) -> list[float]:
        import numpy as np

        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(384)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()


# =============================================================================
# DEPENDENCY PARSER
# =============================================================================


class DependencyParser:
    """Parse import/include statements from code files."""

    PYTHON_STDLIB = {
        "os",
        "sys",
        "re",
        "json",
        "math",
        "datetime",
        "time",
        "random",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "typing",
        "logging",
        "hashlib",
        "subprocess",
        "threading",
        "asyncio",
        "dataclasses",
        "enum",
        "abc",
        "io",
        "copy",
        "pickle",
        "sqlite3",
    }

    def parse(self, content: str, file_path: str, language: str = None) -> list[dict[str, Any]]:
        if not language:
            language = self._detect_language(file_path)

        if language == "python":
            return self._parse_python(content)
        elif language in ("javascript", "typescript"):
            return self._parse_javascript(content)
        return []

    def _detect_language(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
        }.get(ext, "unknown")

    def _parse_python(self, content: str) -> list[dict[str, Any]]:
        import ast

        dependencies = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._parse_python_regex(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(
                        {
                            "module_name": alias.name,
                            "line_number": node.lineno,
                            "is_relative": False,
                            "is_external": alias.name.split(".")[0] not in self.PYTHON_STDLIB,
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                dependencies.append(
                    {
                        "module_name": module,
                        "line_number": node.lineno,
                        "is_relative": node.level > 0,
                        "is_external": (
                            module.split(".")[0] not in self.PYTHON_STDLIB if module else False
                        ),
                    }
                )

        return dependencies

    def _parse_python_regex(self, content: str) -> list[dict[str, Any]]:
        dependencies = []
        import_re = re.compile(r"^import\s+([\w.]+)", re.MULTILINE)
        from_re = re.compile(r"^from\s+(\.*)?([\w.]*)\s+import", re.MULTILINE)

        for i, line in enumerate(content.split("\n"), 1):
            match = import_re.match(line.strip())
            if match:
                dependencies.append(
                    {
                        "module_name": match.group(1),
                        "line_number": i,
                        "is_relative": False,
                        "is_external": match.group(1).split(".")[0] not in self.PYTHON_STDLIB,
                    }
                )
                continue
            match = from_re.match(line.strip())
            if match:
                module = match.group(2) or ""
                dependencies.append(
                    {
                        "module_name": module,
                        "line_number": i,
                        "is_relative": len(match.group(1) or "") > 0,
                        "is_external": (
                            module.split(".")[0] not in self.PYTHON_STDLIB if module else False
                        ),
                    }
                )

        return dependencies

    def _parse_javascript(self, content: str) -> list[dict[str, Any]]:
        dependencies = []
        es6_re = re.compile(r"import\s+.*?\s+from\s+['\"](.+?)['\"]", re.MULTILINE)
        require_re = re.compile(r"require\s*\(\s*['\"](.+?)['\"]", re.MULTILINE)

        for i, line in enumerate(content.split("\n"), 1):
            for pattern in [es6_re, require_re]:
                match = pattern.search(line)
                if match:
                    module = match.group(1)
                    dependencies.append(
                        {
                            "module_name": module,
                            "line_number": i,
                            "is_relative": module.startswith("."),
                            "is_external": not module.startswith("."),
                        }
                    )

        return dependencies


# =============================================================================
# SECURITY SCANNER
# =============================================================================


class SecurityScanner:
    """Scan content for PII and secrets."""

    PII_PATTERNS = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone"),
        (r"\b\d{3}[-]?\d{2}[-]?\d{4}\b", "ssn"),
    ]

    SECRET_PATTERNS = [
        (r'(?:api[_-]?key|apikey)\s*[:=]\s*["\']?[\w-]{20,}', "api_key"),
        (r'(?:secret|password|passwd|pwd)\s*[:=]\s*["\']?[\w!@#$%^&*-]{8,}', "secret"),
        (r"ghp_[a-zA-Z0-9]{36}", "github_token"),
        (r"sk-[a-zA-Z0-9]{48}", "openai_key"),
    ]

    def scan(self, content: str) -> dict[str, list[str]]:
        results = {"pii": [], "secrets": []}
        for pattern, pattern_type in self.PII_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                results["pii"].append(pattern_type)
        for pattern, pattern_type in self.SECRET_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                results["secrets"].append(pattern_type)
        return results


# =============================================================================
# GIT UTILITIES
# =============================================================================


class GitUtil:
    """Git repository utilities."""

    @staticmethod
    def get_repo_info(path: str) -> dict[str, Any] | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"], cwd=path, capture_output=True, text=True
            )
            if result.returncode != 0:
                return None

            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=path,
                capture_output=True,
                text=True,
            )
            remote_url = result.stdout.strip() if result.returncode == 0 else None

            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=path,
                capture_output=True,
                text=True,
            )
            branch = result.stdout.strip() if result.returncode == 0 else "main"

            result = subprocess.run(
                ["git", "log", "-1", "--format=%H|%aI|%s"], cwd=path, capture_output=True, text=True
            )
            commit_info = {}
            if result.returncode == 0:
                parts = result.stdout.strip().split("|", 2)
                if len(parts) == 3:
                    commit_info = {"sha": parts[0], "date": parts[1], "message": parts[2]}

            return {"remote_url": remote_url, "branch": branch, "commit": commit_info}
        except Exception:
            return None

    @staticmethod
    def get_file_commit(path: str, file_path: str) -> dict[str, str] | None:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H|%aI|%an", "--", file_path],
                cwd=path,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("|", 2)
                if len(parts) == 3:
                    return {"sha": parts[0], "date": parts[1], "author": parts[2]}
        except Exception:
            pass
        return None


# =============================================================================
# INGESTION SERVICE
# =============================================================================


class IngestService:
    """Main ingestion service for documents and repositories."""

    def __init__(self, config: IngestConfig = None, db_url: str = None):
        self.config = config or IngestConfig()
        self._engine = None

        self.extractor = TextExtractor()
        self.chunker = TextChunker(
            chunk_tokens=self.config.chunk_tokens, overlap_tokens=self.config.overlap_tokens
        )
        self.embedder = EmbeddingService(model_name=self.config.embedding_model)
        self.dep_parser = DependencyParser()
        self.security_scanner = SecurityScanner()

        if db_url:
            from sqlalchemy import create_engine

            self._engine = create_engine(db_url)

    @property
    def engine(self):
        if self._engine is None:
            self._engine = get_engine()
        return self._engine

    def _get_tenant_id(self, session: Session, tenant_name: str = None) -> str:
        if tenant_name:
            tenant = session.query(Tenant).filter_by(name=tenant_name).first()
            if tenant:
                return str(tenant.id)
        return get_default_tenant_id(session)

    def ingest_document(
        self,
        path: str,
        tenant_name: str = None,
        source: str = "ingestion",
        metadata: dict[str, Any] = None,
    ) -> IngestResult:
        """Ingest a single document."""
        result = IngestResult(success=False)
        start_time = datetime.utcnow()

        try:
            path = Path(path).resolve()

            if not path.exists():
                result.errors.append(f"File not found: {path}")
                return result

            with open(path, "rb") as f:
                content_bytes = f.read()
            content_hash = hashlib.sha256(content_bytes).hexdigest()

            size_mb = len(content_bytes) / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                result.errors.append(f"File too large: {size_mb:.1f}MB")
                return result

            with get_session() as session:
                tenant_id = self._get_tenant_id(session, tenant_name)

                op_inputs = {"path": str(path), "content_hash": content_hash}
                operation, created = get_or_create_operation(
                    session, tenant_id, OperationTypeEnum.INGEST_DOCUMENT, op_inputs
                )
                result.operation_id = str(operation.id)

                if not created and operation.status == OperationStatusEnum.SUCCESS:
                    result.success = True
                    return result

                mark_operation_started(session, operation.id)

                try:
                    text, extract_meta = self.extractor.extract(str(path))
                    if not text:
                        raise ValueError("No text extracted")

                    security_flags = (
                        self.security_scanner.scan(text) if self.config.scan_for_pii else {}
                    )

                    resource = Resource(
                        tenant_id=tenant_id,
                        file_path=str(path),
                        file_name=path.name,
                        file_extension=path.suffix,
                        resource_type=ResourceTypeEnum.DOCUMENT,
                        content_hash=content_hash,
                        size_bytes=len(content_bytes),
                        line_count=text.count("\n") + 1,
                        token_count=TextExtractor.count_tokens(text),
                        has_pii=bool(security_flags.get("pii")),
                        has_secrets=bool(security_flags.get("secrets")),
                        is_parsed=True,
                        metadata=extract_meta,
                    )
                    session.add(resource)
                    session.flush()
                    result.resources_created = 1

                    document = Document(
                        tenant_id=tenant_id,
                        source_resource_id=resource.id,
                        title=path.stem,
                        content=text,
                        content_hash=compute_content_hash(text),
                        doc_type=KnowledgeTypeEnum.DOCUMENTATION,
                        source=source,
                        word_count=len(text.split()),
                        token_count=TextExtractor.count_tokens(text),
                        has_pii=resource.has_pii,
                        tags=[path.suffix.lstrip(".")],
                        metadata={**(metadata or {}), **extract_meta},
                    )
                    session.add(document)
                    session.flush()
                    result.documents_created = 1

                    chunks = self.chunker.chunk(text)
                    chunk_texts = [c["content"] for c in chunks]

                    embeddings = []
                    if self.config.generate_embeddings and chunk_texts:
                        embeddings = self.embedder.embed(
                            chunk_texts, batch_size=self.config.embedding_batch_size
                        )
                        result.embeddings_generated = len(embeddings)

                    for i, chunk_data in enumerate(chunks):
                        chunk = DocChunk(
                            tenant_id=tenant_id,
                            document_id=document.id,
                            content=chunk_data["content"],
                            chunk_index=chunk_data["index"],
                            start_char=chunk_data.get("start_char"),
                            end_char=chunk_data.get("end_char"),
                            token_count=chunk_data["token_count"],
                            char_count=chunk_data["char_count"],
                            embedding=embeddings[i] if embeddings else None,
                            embedding_model=self.config.embedding_model if embeddings else None,
                            embedded_at=datetime.utcnow() if embeddings else None,
                        )
                        session.add(chunk)

                    result.chunks_created = len(chunks)
                    document.is_chunked = True
                    document.is_embedded = bool(embeddings)

                    mark_operation_success(
                        session,
                        operation.id,
                        {"document_id": str(document.id), "chunks": len(chunks)},
                    )
                    result.success = True

                except Exception as e:
                    result.errors.append(str(e))
                    mark_operation_failed(session, operation.id, str(e), traceback.format_exc())
                    session.rollback()

        except Exception as e:
            result.errors.append(f"Ingestion failed: {e}")

        finally:
            result.duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return result

    def ingest_repository(
        self, path: str, url: str = None, tenant_name: str = None, source: str = "repo_ingestion"
    ) -> IngestResult:
        """Ingest a Git repository."""
        result = IngestResult(success=False)
        start_time = datetime.utcnow()

        try:
            repo_path = Path(path).resolve()

            if not repo_path.exists():
                result.errors.append(f"Repository not found: {path}")
                return result

            git_info = GitUtil.get_repo_info(str(repo_path))

            with get_session() as session:
                tenant_id = self._get_tenant_id(session, tenant_name)

                op_inputs = {
                    "path": str(repo_path),
                    "url": url or (git_info or {}).get("remote_url"),
                }
                operation, created = get_or_create_operation(
                    session, tenant_id, OperationTypeEnum.INGEST_REPO, op_inputs
                )
                result.operation_id = str(operation.id)

                mark_operation_started(session, operation.id)

                try:
                    repo_url = url or (git_info or {}).get("remote_url") or str(repo_path)

                    repository = (
                        session.query(Repository)
                        .filter_by(tenant_id=tenant_id, url=repo_url)
                        .first()
                    )
                    if not repository:
                        repository = Repository(
                            tenant_id=tenant_id,
                            name=repo_path.name,
                            url=repo_url,
                            clone_path=str(repo_path),
                            default_branch=(git_info or {}).get("branch", "main"),
                        )
                        session.add(repository)
                        session.flush()

                    if git_info and git_info.get("commit"):
                        commit = git_info["commit"]
                        repository.last_commit_sha = commit.get("sha")
                        repository.last_commit_date = (
                            datetime.fromisoformat(commit["date"]) if commit.get("date") else None
                        )
                        repository.last_commit_message = commit.get("message")

                    repository.last_ingested_at = datetime.utcnow()

                    error_count = 0
                    file_count = 0

                    for file_path in self._iter_repo_files(repo_path):
                        if error_count >= self.config.max_errors_per_batch:
                            break

                        try:
                            file_result = self._process_repo_file(
                                session, tenant_id, repository, repo_path, file_path, source
                            )
                            result.resources_created += file_result.get("resources", 0)
                            result.documents_created += file_result.get("documents", 0)
                            result.chunks_created += file_result.get("chunks", 0)
                            result.embeddings_generated += file_result.get("embeddings", 0)
                            result.dependencies_found += file_result.get("dependencies", 0)
                            file_count += 1
                        except Exception as e:
                            if self.config.continue_on_error:
                                error_count += 1
                                result.errors.append(f"{file_path}: {e}")
                            else:
                                raise

                    repository.file_count = file_count
                    mark_operation_success(
                        session,
                        operation.id,
                        {"repository_id": str(repository.id), "files": file_count},
                    )
                    result.success = True

                except Exception as e:
                    result.errors.append(str(e))
                    mark_operation_failed(session, operation.id, str(e), traceback.format_exc())
                    session.rollback()

        except Exception as e:
            result.errors.append(f"Repository ingestion failed: {e}")

        finally:
            result.duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return result

    def _iter_repo_files(self, repo_path: Path) -> Generator[Path, None, None]:
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self.config.include_extensions:
                continue

            rel_path = str(file_path.relative_to(repo_path))
            skip = any(
                p in rel_path if not p.startswith("*") else rel_path.endswith(p[1:])
                for p in self.config.exclude_patterns
            )
            if not skip:
                yield file_path

    def _process_repo_file(
        self,
        session: Session,
        tenant_id: str,
        repository: Repository,
        repo_path: Path,
        file_path: Path,
        source: str,
    ) -> dict[str, int]:
        counters = {"resources": 0, "documents": 0, "chunks": 0, "embeddings": 0, "dependencies": 0}
        rel_path = str(file_path.relative_to(repo_path))

        with open(file_path, "rb") as f:
            content_bytes = f.read()
        content_hash = hashlib.sha256(content_bytes).hexdigest()

        existing = (
            session.query(Resource)
            .filter_by(tenant_id=tenant_id, repository_id=repository.id, file_path=rel_path)
            .first()
        )
        if existing and existing.content_hash == content_hash:
            return counters

        try:
            text, extract_meta = self.extractor.extract(str(file_path))
        except Exception:
            return counters

        security_flags = self.security_scanner.scan(text) if self.config.scan_for_pii else {}
        commit_info = GitUtil.get_file_commit(str(repo_path), rel_path)
        ext = file_path.suffix.lower()

        resource_type = {
            ".py": ResourceTypeEnum.CODE_FILE,
            ".js": ResourceTypeEnum.CODE_FILE,
            ".pdf": ResourceTypeEnum.DOCUMENT,
            ".md": ResourceTypeEnum.DOCUMENT,
        }.get(ext, ResourceTypeEnum.CODE_FILE)

        if existing:
            resource = existing
            resource.content_hash = content_hash
        else:
            resource = Resource(
                tenant_id=tenant_id,
                repository_id=repository.id,
                file_path=rel_path,
                file_name=file_path.name,
                file_extension=ext,
                resource_type=resource_type,
                content_hash=content_hash,
            )
            session.add(resource)

        resource.size_bytes = len(content_bytes)
        resource.line_count = text.count("\n") + 1
        resource.token_count = TextExtractor.count_tokens(text)
        resource.language = ext.lstrip(".")
        resource.has_pii = bool(security_flags.get("pii"))
        resource.has_secrets = bool(security_flags.get("secrets"))
        resource.is_parsed = True

        if commit_info:
            resource.last_commit_sha = commit_info.get("sha")
            resource.last_modified_at = (
                datetime.fromisoformat(commit_info["date"]) if commit_info.get("date") else None
            )

        session.flush()
        counters["resources"] = 1

        if self.config.parse_dependencies and ext in [".py", ".js", ".ts", ".jsx", ".tsx"]:
            for dep in self.dep_parser.parse(text, rel_path):
                session.add(
                    ResourceDependency(
                        tenant_id=tenant_id,
                        source_id=resource.id,
                        dependency_type="import",
                        import_path=dep.get("module_name"),
                        line_number=dep.get("line_number"),
                    )
                )
                counters["dependencies"] += 1

        doc_type = (
            KnowledgeTypeEnum.CODE_SNIPPET
            if resource_type == ResourceTypeEnum.CODE_FILE
            else KnowledgeTypeEnum.DOCUMENTATION
        )
        document = Document(
            tenant_id=tenant_id,
            source_resource_id=resource.id,
            title=file_path.name,
            content=text,
            content_hash=compute_content_hash(text),
            doc_type=doc_type,
            source=source,
            word_count=len(text.split()),
            token_count=resource.token_count,
            has_pii=resource.has_pii,
            tags=[ext.lstrip("."), "repository"],
            metadata={"repository_id": str(repository.id), "file_path": rel_path, **extract_meta},
        )
        session.add(document)
        session.flush()
        counters["documents"] = 1

        chunks = self.chunker.chunk(text)
        chunk_texts = [c["content"] for c in chunks]
        embeddings = (
            self.embedder.embed(chunk_texts, batch_size=self.config.embedding_batch_size)
            if self.config.generate_embeddings and chunk_texts
            else []
        )
        counters["embeddings"] = len(embeddings)

        for i, chunk_data in enumerate(chunks):
            session.add(
                DocChunk(
                    tenant_id=tenant_id,
                    document_id=document.id,
                    content=chunk_data["content"],
                    chunk_index=chunk_data["index"],
                    start_char=chunk_data.get("start_char"),
                    end_char=chunk_data.get("end_char"),
                    token_count=chunk_data["token_count"],
                    char_count=chunk_data["char_count"],
                    embedding=embeddings[i] if embeddings else None,
                    embedding_model=self.config.embedding_model if embeddings else None,
                    embedded_at=datetime.utcnow() if embeddings else None,
                )
            )

        counters["chunks"] = len(chunks)
        document.is_chunked = True
        document.is_embedded = bool(embeddings)

        return counters


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """Command-line entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Princeps Brain Layer Ingestion Service")
    parser.add_argument("path", help="Path to document or repository")
    parser.add_argument("--type", choices=["document", "repository"], default="document")
    parser.add_argument("--url", help="Repository URL")
    parser.add_argument("--tenant", help="Tenant name")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--chunk-tokens", type=int, default=800)

    args = parser.parse_args()

    config = IngestConfig(
        chunk_tokens=args.chunk_tokens, generate_embeddings=not args.no_embeddings
    )
    service = IngestService(config=config)

    if args.type == "document":
        result = service.ingest_document(args.path, tenant_name=args.tenant)
    else:
        result = service.ingest_repository(args.path, url=args.url, tenant_name=args.tenant)

    print(f"{'SUCCESS' if result.success else 'FAILED'}")
    print(
        f"Documents: {result.documents_created}, Chunks: {result.chunks_created}, Embeddings: {result.embeddings_generated}"
    )


if __name__ == "__main__":
    main()
