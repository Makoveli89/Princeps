#!/usr/bin/env python3
# pylint: disable=invalid-name
"""
Vector store abstraction with a Chroma on-disk backend.

- Uses sentence-transformers if available for embeddings
- Falls back to a simple hashing-based embedding (no extra deps)
- If chromadb is not installed, the store runs in disabled mode and no-ops
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Optional imports
try:
    import chromadb  # type: ignore
    try:
        from chromadb.config import Settings  # type: ignore
    except Exception:  # pragma: no cover
        Settings = None  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None  # type: ignore
    Settings = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


@dataclass
class VectorResult:
    id: str
    score: float
    metadata: Dict[str, Any]
    document: str


class BaseEmbedding:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class SbertEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self.model = SentenceTransformer(model_name)  # type: ignore[operator]

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()  # type: ignore[no-any-return]


class HashingEmbedding(BaseEmbedding):
    """Simple hashing-based embedding with fixed dimension, no deps.

    Not semantically strong, but stable and sufficient to let Chroma
    operate locally without external ML packages.
    """

    def __init__(self, dim: int = 384, seed: int = 13) -> None:
        self.dim = dim
        self.seed = seed

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    def _hash(self, token: str) -> int:
        h = hashlib.md5((token + str(self.seed)).encode("utf-8")).hexdigest()
        return int(h, 16)

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = self._tokenize(text)
            if not tokens:
                out.append(vec)
                continue
            for tok in tokens:
                idx = self._hash(tok) % self.dim
                vec[idx] += 1.0
            # L2 normalize
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            out.append(vec)
        return out


def build_embedding(prefer_sbert: bool = True) -> BaseEmbedding:
    if prefer_sbert:
        try:
            return SbertEmbedding()
        except Exception:
            pass
    return HashingEmbedding()


class ChromaVectorStore:
    """Chroma-backed vector store with on-disk persistence.

    Disabled gracefully if chromadb is not installed or VECTOR_STORE_DISABLED=1.
    """

    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection_name: str = "mothership_learnings",
        prefer_sbert: bool = True,
    ) -> None:
        self.enabled = bool(chromadb) and os.getenv("VECTOR_STORE_DISABLED", "0") != "1"
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding = build_embedding(prefer_sbert=prefer_sbert)
        self._client = None
        self._collection = None
        if self.enabled:
            self._init_client()

    def _init_client(self) -> None:
        assert chromadb is not None and Settings is not None

        self._client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings())
        # We compute embeddings ourselves and pass them on upsert/query
        self._collection = self._client.get_or_create_collection(self.collection_name)

    def is_available(self) -> bool:
        return bool(self.enabled)

    @staticmethod
    def _pack_content(task: str, feedback: str, solution: Optional[Dict[str, Any]] = None) -> str:
        sol = ""
        if solution:
            try:
                sol = "\nSolution: " + str(solution)
            except Exception:
                sol = ""
        return f"Task: {task}\nFeedback: {feedback}{sol}"

    def upsert_learning(
        self,
        learning_id: str,
        task: str,
        feedback: str,
        solution: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.enabled or not self._collection:
            return False
        doc = self._pack_content(task, feedback, solution)
        emb = self.embedding.embed([doc])[0]
        try:
            self._collection.upsert(
                ids=[learning_id],
                documents=[doc],
                metadatas=[metadata or {}],
                embeddings=[emb],
            )
            return True
        except Exception:
            return False

    def query(self, query_text: str, top_k: int = 5) -> List[VectorResult]:
        if not self.enabled or not self._collection:
            return []
        qemb = self.embedding.embed([query_text])[0]
        try:
            res = self._collection.query(query_embeddings=[qemb], n_results=top_k)
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]  # cosine distance by default
            out: List[VectorResult] = []
            for i, _id in enumerate(ids):
                # Convert distance to a similarity-ish score
                dist = dists[i] if i < len(dists) else None
                score = 1.0 - float(dist) if dist is not None else 0.0
                out.append(VectorResult(id=_id, score=score, metadata=metas[i] or {}, document=docs[i]))
            return out
        except Exception:
            return []
