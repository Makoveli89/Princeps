"""
Embedding-based retrieval with multiple backend support.

Backends (in order of preference):
- sbert: Sentence-BERT embeddings (requires sentence-transformers + numpy)
- tfidf: Lightweight TF-IDF (no external deps)
- heuristic: Token overlap / Jaccard similarity (no deps)

Each backend falls back gracefully if dependencies aren't available.
"""

import math
from collections import Counter
from typing import Any, Dict, List, cast


class HeuristicRetriever:
    """Simple heuristic retrieval using token overlap / Jaccard-like score."""

    @staticmethod
    def tokenize(text: str):
        return [t.strip().lower() for t in text.split() if t.strip()]

    @staticmethod
    def score(query: str, doc: str) -> float:
        q = set(HeuristicRetriever.tokenize(query))
        d = set(HeuristicRetriever.tokenize(doc))
        if not q or not d:
            return 0.0
        inter = q & d
        union = q | d
        return len(inter) / len(union)

    def retrieve(self, query: str, docs: List[Dict[str, Any]], top_k=5):
        scored = []
        for d in docs:
            s = self.score(query, d.get("text", ""))
            scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for s, doc in scored[:top_k]]


class SimpleTfidfRetriever:
    """Lightweight TF-IDF retriever without external deps.
    Note: This is a simple educational implementation — not optimized.
    """

    def __init__(self, docs: List[Dict[str, Any]]):
        texts = [d.get("text", "") for d in docs]
        self.docs = docs
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []
        self._build_vocab(texts)

    def _build_vocab(self, texts: List[str]):
        df = Counter()
        for text in texts:
            tokens = set(self._tokenize(text))
            for t in tokens:
                df[t] += 1
        # build vocab
        for i, t in enumerate(sorted(df.keys())):
            self.vocab[t] = i
        N = max(1, len(texts))
        for t, cnt in df.items():
            self.idf[t] = math.log(N / (1 + cnt)) + 1.0
        # build doc vectors
        for text in texts:
            vec = [0.0] * len(self.vocab)
            tf = Counter(self._tokenize(text))
            for token, f in tf.items():
                if token in self.vocab:
                    vec[self.vocab[token]] = f * self.idf.get(token, 0.0)
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            self.doc_vectors.append(vec)

    @staticmethod
    def _tokenize(text: str):
        return [t.strip().lower() for t in text.split() if t.strip()]

    @staticmethod
    def _dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def retrieve(self, query: str, top_k=5):
        qtf = Counter(self._tokenize(query))
        qvec = [0.0] * len(self.vocab)
        for token, f in qtf.items():
            if token in self.vocab:
                qvec[self.vocab[token]] = f * self.idf.get(token, 0.0)
        qnorm = math.sqrt(sum(x * x for x in qvec))
        if qnorm > 0:
            qvec = [x / qnorm for x in qvec]
        scores = []
        for vec, doc in zip(self.doc_vectors, self.docs):
            if len(vec) != len(qvec):
                # inconsistent vocab; score 0
                scores.append((0.0, doc))
            else:
                scores.append((self._dot(qvec, vec), doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for s, doc in scores[:top_k]]


class EmbeddingRetriever:
    """Facade that exposes heuristic, tfidf, and optional SBERT backends.

    Backends:
    - heuristic: token-overlap scoring (no deps)
    - tfidf: lightweight TF-IDF (no external deps)
    - sbert: uses sentence-transformers if available; falls back gracefully
    """

    def __init__(self, docs: List[Dict[str, Any]], backend="tfidf"):
        self.docs = docs
        self.backend_name = backend
        if backend == "sbert":
            try:
                # Lazy import to avoid hard dependency
                import numpy as np  # type: ignore[import-not-found]
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

                class _SbertRetriever:
                    def __init__(self, docs):
                        self.docs = docs
                        self.model = SentenceTransformer("all-MiniLM-L6-v2")
                        self.doc_texts = [d.get("text", "") for d in docs]
                        self.embs = self.model.encode(self.doc_texts, show_progress_bar=False)

                    @staticmethod
                    def _cos_sim(a, b):
                        # a: (D,), b: (N,D)
                        denom = np.linalg.norm(a) * np.linalg.norm(b, axis=1) + 1e-12
                        return (b @ a) / denom

                    def retrieve(self, query: str, top_k=5):
                        q = self.model.encode([query], show_progress_bar=False)[0]
                        sims = self._cos_sim(q, self.embs)
                        idxs = sims.argsort()[::-1][:top_k]
                        return [self.docs[i] for i in idxs]

                self.engine = _SbertRetriever(docs)
                self.backend_name = "sbert"
                return
            except Exception:
                # Fallback preferences: TF-IDF then heuristic
                try:
                    self.engine = SimpleTfidfRetriever(docs)
                    self.backend_name = "tfidf"
                except Exception:
                    self.engine = HeuristicRetriever()
                    self.backend_name = "heuristic"
                return

        if backend == "tfidf":
            try:
                self.engine = SimpleTfidfRetriever(docs)
                self.backend_name = "tfidf"
                return
            except Exception:
                self.engine = HeuristicRetriever()
                self.backend_name = "heuristic"
                return

        # Default: heuristic
        self.engine = HeuristicRetriever()
        self.backend_name = "heuristic"

    def retrieve(self, query: str, top_k=5):
        if isinstance(self.engine, SimpleTfidfRetriever):
            return self.engine.retrieve(query, top_k=top_k)
        if self.backend_name == "sbert":
            # SBERT retriever has a uniform interface
            return self.engine.retrieve(query, top_k=top_k)
        if isinstance(self.engine, HeuristicRetriever):
            # heuristic engine expects (query, docs, top_k)
            hr = cast(HeuristicRetriever, self.engine)
            return hr.retrieve(query, self.docs, top_k)  # type: ignore[call-arg]
        # Fallback: try heuristic-style
        try:
            hr = cast(HeuristicRetriever, self.engine)
            return hr.retrieve(query, self.docs, top_k)  # type: ignore[call-arg]
        except Exception:
            return []
