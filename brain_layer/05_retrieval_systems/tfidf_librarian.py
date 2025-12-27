"""
Librarian: lightweight knowledge service for ingesting repository docs/code
and providing TF-IDF based retrieval (no heavy dependencies).

Features:
- Ingest: globs or raw texts; chunks text; builds TF-IDF index
- Persist: saves index to JSON under data/knowledge_index/
- Query: returns top-k relevant chunks with scores

Intended for RAG-style context injection to coding agents.

Source: Lumina/src/services/librarian.py
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return TOKEN_RE.findall(text)


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j]
        chunks.append(chunk)
        if j >= n:
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks


@dataclass
class Chunk:
    id: str
    source: str
    start: int
    text: str
    tokens: Dict[str, float]  # tf-idf weights
    norm: float


class TfidfLibrarian:
    """Simple TF-IDF knowledge base builder and retriever.
    
    No external dependencies required. Persists index to JSON.
    Good as a fallback when vector stores aren't available.
    """

    def __init__(self, project_root: str, storage_dir: Optional[str] = None):
        self.project_root = project_root
        self.storage_dir = storage_dir or os.path.join(project_root, "data", "knowledge_index")
        os.makedirs(self.storage_dir, exist_ok=True)

        self.docfreq: Dict[str, int] = {}
        self.total_docs: int = 0
        self.chunks: List[Chunk] = []
        self.index_path = os.path.join(self.storage_dir, "index.json")
        self.meta_path = os.path.join(self.storage_dir, "meta.json")

        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.docfreq = meta.get("docfreq", {})
                self.total_docs = int(meta.get("total_docs", 0))
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.chunks = [Chunk(**c) for c in data]
        except Exception:
            # Corrupt index; start fresh
            self.docfreq = {}
            self.total_docs = 0
            self.chunks = []

    def _save(self) -> None:
        meta = {
            "docfreq": self.docfreq,
            "total_docs": self.total_docs,
            "updated_at": time.time(),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)

    # ---------- TF-IDF helpers ----------
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        if not tf:
            return tf
        inv_n = 1.0 / sum(tf.values())
        for k in list(tf.keys()):
            tf[k] *= inv_n
        return tf

    def _idf(self, term: str) -> float:
        df = self.docfreq.get(term, 0)
        # Smooth idf
        return math.log((1 + self.total_docs) / (1 + df)) + 1.0

    def _tfidf(self, tokens: List[str]) -> Tuple[Dict[str, float], float]:
        tf = self._compute_tf(tokens)
        weights: Dict[str, float] = {}
        for t, val in tf.items():
            weights[t] = val * self._idf(t)
        # L2 norm
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        return weights, norm

    def _update_docfreq(self, doc_terms: Iterable[str]) -> None:
        seen = set(doc_terms)
        for t in seen:
            self.docfreq[t] = self.docfreq.get(t, 0) + 1

    # ---------- ingestion ----------
    def ingest_paths(
        self,
        patterns: List[str],
        max_file_kb: int = 256,
        exts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        exts = exts or [
            ".md",
            ".py",
            ".ts",
            ".tsx",
            ".js",
            ".json",
            ".yml",
            ".yaml",
            ".ps1",
        ]
        files: List[str] = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(self.project_root, pat), recursive=True))

        files = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in exts]
        files = list(dict.fromkeys(files))  # dedupe

        added = 0
        for fp in files:
            try:
                if os.path.getsize(fp) > max_file_kb * 1024:
                    continue
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                rel = os.path.relpath(fp, self.project_root)
                chunks = chunk_text(text)
                for idx, ch in enumerate(chunks):
                    toks = tokenize(ch)
                    self._update_docfreq(toks)
                    self.total_docs += 1
                    w, n = self._tfidf(toks)
                    self.chunks.append(
                        Chunk(
                            id=f"{rel}::chunk{idx}",
                            source=rel,
                            start=idx,
                            text=ch,
                            tokens=w,
                            norm=n,
                        )
                    )
                    added += 1
            except Exception:
                continue

        # Recompute weights with updated idf for consistency
        if added > 0:
            for c in self.chunks:
                w, n = self._tfidf(list(c.tokens.keys()))
                c.tokens, c.norm = w, n
            self._save()

        return {"files_scanned": len(files), "chunks_added": added}

    def ingest_texts(self, items: List[Dict[str, str]]) -> Dict[str, Any]:
        """Ingest raw texts; each item: {"id": str, "text": str}"""
        added = 0
        for it in items:
            text = it.get("text", "")
            sid = it.get("id") or f"raw::{len(self.chunks)}"
            for idx, ch in enumerate(chunk_text(text)):
                toks = tokenize(ch)
                self._update_docfreq(toks)
                self.total_docs += 1
                w, n = self._tfidf(toks)
                self.chunks.append(
                    Chunk(
                        id=f"{sid}::chunk{idx}",
                        source=sid,
                        start=idx,
                        text=ch,
                        tokens=w,
                        norm=n,
                    )
                )
                added += 1

        if added > 0:
            for c in self.chunks:
                w, n = self._tfidf(list(c.tokens.keys()))
                c.tokens, c.norm = w, n
            self._save()
        return {"chunks_added": added}

    # ---------- query ----------
    def query(self, text: str, top_k: int = 4, min_score: float = 0.05) -> List[Dict[str, Any]]:
        q_tokens = tokenize(text)
        q_w, q_norm = self._tfidf(q_tokens)
        results: List[Tuple[float, Chunk]] = []
        for c in self.chunks:
            # sparse dot product over shared terms
            score = 0.0
            for t, qw in q_w.items():
                cw = c.tokens.get(t)
                if cw:
                    score += qw * cw
            score = score / (q_norm * c.norm)
            if score >= min_score:
                results.append((score, c))
        results.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, ch in results[:top_k]:
            out.append(
                {
                    "id": ch.id,
                    "source": ch.source,
                    "start": ch.start,
                    "score": s,
                    "text": ch.text,
                }
            )
        return out

    # ---------- convenience ----------
    def auto_ingest_minimal(self) -> Dict[str, Any]:
        patterns = [
            "README.md",
            "docs/**/*.md",
            "src/**/*.py",
            "src/**/*.ts",
            "src/**/*.tsx",
            "scripts/**/*.ps1",
            ".github/**/*.yml",
        ]
        return self.ingest_paths(patterns)
    
    def clear(self) -> None:
        """Clear all indexed chunks and reset state."""
        self.chunks = []
        self.docfreq = {}
        self.total_docs = 0
        self._save()
    
    def stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        return {
            "total_chunks": len(self.chunks),
            "total_docs": self.total_docs,
            "vocab_size": len(self.docfreq),
            "storage_dir": self.storage_dir,
        }
