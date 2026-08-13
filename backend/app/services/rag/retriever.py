"""BM25 + 벡터(앙상블) 검색기 — Ollama 임베딩 사용."""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings
from app.services.rag.documents import AnalysisDocument
from app.services.rag.llm_router import get_llm_router

log = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    # 한글/영문/숫자 토큰
    return re.findall(r"[a-z0-9]+|[가-힣]+", text)


class RAGRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.documents: List[AnalysisDocument] = []
        self._doc_tokens: List[List[str]] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0
        self._embeddings: Optional[np.ndarray] = None
        self._ready = False

    def build(self, documents: List[AnalysisDocument]) -> str:
        self.documents = documents or []
        if not self.documents:
            self._ready = False
            return "no documents"

        self._doc_tokens = [_tokenize(d.page_content) for d in self.documents]
        self._df = Counter()
        total_len = 0
        for tokens in self._doc_tokens:
            total_len += len(tokens)
            for term in set(tokens):
                self._df[term] += 1
        self._avgdl = (total_len / len(self._doc_tokens)) if self._doc_tokens else 0.0

        mode = "BM25"
        try:
            router = get_llm_router()
            # 문서 인덱싱은 passage, 질의는 query (NVIDIA NeMo Retriever 규약)
            vectors = router.embed(
                [d.page_content[:2000] for d in self.documents],
                input_type="passage",
            )
            self._embeddings = np.array(vectors, dtype=np.float32)
            # L2 normalize
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._embeddings = self._embeddings / norms
            mode = f"BM25 + FAISS-like ({self.settings.embed_provider} embeddings)"
        except Exception as e:
            log.warning("embedding index failed, BM25 only: %s", e)
            self._embeddings = None

        self._ready = True
        return mode

    def _bm25_scores(self, query: str) -> np.ndarray:
        q_tokens = _tokenize(query)
        n = len(self.documents)
        scores = np.zeros(n, dtype=np.float32)
        k1, b = 1.5, 0.75
        for i, tokens in enumerate(self._doc_tokens):
            if not tokens:
                continue
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                freq = tf[term]
                score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / (self._avgdl or 1)))
            scores[i] = score
        return scores

    def _vector_scores(self, query: str) -> np.ndarray:
        n = len(self.documents)
        if self._embeddings is None:
            return np.zeros(n, dtype=np.float32)
        try:
            q = np.array(
                get_llm_router().embed_one(query[:2000], input_type="query"),
                dtype=np.float32,
            )
            norm = np.linalg.norm(q)
            if norm == 0:
                return np.zeros(n, dtype=np.float32)
            q = q / norm
            return self._embeddings @ q
        except Exception as e:
            log.warning("vector search failed: %s", e)
            return np.zeros(n, dtype=np.float32)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[AnalysisDocument]:
        if not self._ready or not self.documents:
            return []
        top_k = top_k or max(self.settings.bm25_k, self.settings.faiss_k)
        bm25 = self._bm25_scores(query)
        vec = self._vector_scores(query)

        def _norm(arr: np.ndarray) -> np.ndarray:
            mn, mx = float(arr.min()), float(arr.max())
            if mx - mn < 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        combined = (
            self.settings.bm25_weight * _norm(bm25)
            + self.settings.faiss_weight * _norm(vec)
        )
        order = np.argsort(-combined)[: top_k * 2]
        selected = self._select_diverse([self.documents[i] for i in order], self.settings.max_docs)
        return selected

    def _select_diverse(self, docs: List[AnalysisDocument], max_docs: int) -> List[AnalysisDocument]:
        counts: Dict[str, int] = {}
        selected: List[AnalysisDocument] = []
        max_per = self.settings.max_doc_types_per_type
        for doc in docs:
            dtype = str(doc.metadata.get("type", "unknown"))
            if counts.get(dtype, 0) >= max_per:
                continue
            selected.append(doc)
            counts[dtype] = counts.get(dtype, 0) + 1
            if len(selected) >= max_docs:
                break
        selected.sort(key=lambda d: d.metadata.get("date", ""), reverse=True)
        return selected

    def get_context(self, query: str) -> Tuple[str, List[AnalysisDocument]]:
        docs = self.retrieve(query)
        parts: List[str] = []
        used: List[AnalysisDocument] = []
        length = 0
        max_len = self.settings.max_context_length
        for doc in docs:
            content = doc.page_content
            if length + len(content) > max_len:
                remain = max_len - length
                if remain > 100:
                    parts.append(content[: remain - 3] + "...")
                    used.append(doc)
                break
            parts.append(content)
            used.append(doc)
            length += len(content) + 2
        return "\n\n".join(parts).strip(), used
