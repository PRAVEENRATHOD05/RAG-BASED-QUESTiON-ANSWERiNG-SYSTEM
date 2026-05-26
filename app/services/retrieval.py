from __future__ import annotations

import math
import re
from collections import Counter

from app.models.entities import DocumentChunk, SearchHit
from app.services.vector_store import LocalVectorStore

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_+-]{1,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "about",
    "from",
    "into",
    "what",
    "when",
    "where",
    "which",
    "tell",
    "please",
    "resume",
    "profile",
    "about",
    "me",
}


class HybridRetriever:
    """Hybrid vector + BM25 retrieval with lightweight reranking and deduping."""

    def __init__(
        self,
        vector_store: LocalVectorStore,
        vector_weight: float = 0.68,
        lexical_weight: float = 0.32,
    ) -> None:
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight

    def retrieve(
        self,
        question: str,
        query_vector,
        top_k: int,
        min_score: float,
    ) -> list[SearchHit]:
        all_vector_scores = self.vector_store.score_all(query_vector)
        if not all_vector_scores:
            return []

        chunks = [chunk for chunk, _ in all_vector_scores]
        vector_scores_by_id = {chunk.id: score for chunk, score in all_vector_scores}
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        lexical_scores_by_id = self._bm25_scores(question=question, chunks=chunks)

        vector_top = max(top_k * 4, 16)
        lexical_top = max(top_k * 4, 16)

        vector_candidate_ids = [chunk.id for chunk, _ in all_vector_scores[:vector_top]]
        lexical_ranked = sorted(
            lexical_scores_by_id.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        lexical_candidate_ids = [chunk_id for chunk_id, _ in lexical_ranked[:lexical_top] if _ > 0]

        candidate_ids = list(dict.fromkeys(vector_candidate_ids + lexical_candidate_ids))
        if not candidate_ids:
            candidate_ids = vector_candidate_ids[:top_k]

        vector_values = [vector_scores_by_id.get(chunk_id, 0.0) for chunk_id in candidate_ids]
        lexical_values = [lexical_scores_by_id.get(chunk_id, 0.0) for chunk_id in candidate_ids]
        norm_vector = self._normalize(vector_values, clamp_low=-0.35, clamp_high=0.95)
        norm_lexical = self._normalize(lexical_values, clamp_low=0.0, clamp_high=max(lexical_values) if lexical_values else 1.0)
        query_tokens = set(self._tokenize(question))

        candidates: list[SearchHit] = []
        for index, chunk_id in enumerate(candidate_ids):
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                continue

            vector_score = vector_scores_by_id.get(chunk_id, 0.0)
            lexical_score = lexical_scores_by_id.get(chunk_id, 0.0)
            overlap = self._token_overlap(query_tokens, chunk.text)
            rerank_bonus = 0.08 if overlap >= 2 else 0.0
            if overlap >= 4:
                rerank_bonus = 0.16

            blended = (
                self.vector_weight * norm_vector[index]
                + self.lexical_weight * norm_lexical[index]
                + rerank_bonus
            )
            candidates.append(
                SearchHit(
                    chunk=chunk,
                    score=round(float(blended), 6),
                    vector_score=round(float(vector_score), 6),
                    lexical_score=round(float(lexical_score), 6),
                )
            )

        deduped = self._dedupe_hits(candidates)
        filtered = [hit for hit in deduped if hit.score >= min_score or hit.lexical_score > 0]

        ranked = sorted(
            filtered if filtered else deduped,
            key=lambda hit: (hit.score, hit.vector_score, hit.lexical_score),
            reverse=True,
        )
        return ranked[:top_k]

    def _bm25_scores(self, question: str, chunks: list[DocumentChunk]) -> dict[str, float]:
        tokens = [token for token in self._tokenize(question) if token not in STOPWORDS]
        if not tokens:
            return {chunk.id: 0.0 for chunk in chunks}

        token_counts_by_chunk: dict[str, Counter[str]] = {}
        doc_lengths: dict[str, int] = {}
        df: Counter[str] = Counter()

        for chunk in chunks:
            words = self._tokenize(chunk.text)
            counts = Counter(words)
            token_counts_by_chunk[chunk.id] = counts
            doc_lengths[chunk.id] = len(words)
            for token in set(tokens):
                if counts[token] > 0:
                    df[token] += 1

        avg_doc_len = sum(doc_lengths.values()) / max(len(doc_lengths), 1)
        k1 = 1.35
        b = 0.72
        total_docs = len(chunks)
        scores: dict[str, float] = {}

        for chunk in chunks:
            counts = token_counts_by_chunk[chunk.id]
            doc_len = doc_lengths[chunk.id]
            score = 0.0

            for token in tokens:
                freq = counts[token]
                if freq <= 0:
                    continue
                idf = math.log(((total_docs - df[token] + 0.5) / (df[token] + 0.5)) + 1.0)
                denom = freq + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1e-6)))
                score += idf * ((freq * (k1 + 1)) / max(denom, 1e-6))

            scores[chunk.id] = score

        return scores

    def _dedupe_hits(self, hits: list[SearchHit]) -> list[SearchHit]:
        deduped: list[SearchHit] = []
        seen_signatures: set[str] = set()

        for hit in sorted(hits, key=lambda item: item.score, reverse=True):
            signature = self._signature(hit.chunk.text)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped.append(hit)
        return deduped

    def _signature(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip().lower()
        return normalized[:220]

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text.lower())]

    def _token_overlap(self, query_tokens: set[str], text: str) -> int:
        if not query_tokens:
            return 0
        text_tokens = set(self._tokenize(text))
        return len(query_tokens & text_tokens)

    def _normalize(
        self,
        values: list[float],
        clamp_low: float | None = None,
        clamp_high: float | None = None,
    ) -> list[float]:
        if not values:
            return []

        processed = values[:]
        if clamp_low is not None or clamp_high is not None:
            low = clamp_low if clamp_low is not None else min(processed)
            high = clamp_high if clamp_high is not None else max(processed)
            processed = [min(max(value, low), high) for value in processed]

        minimum = min(processed)
        maximum = max(processed)
        if maximum - minimum < 1e-9:
            return [0.5 if value > 0 else 0.0 for value in processed]

        return [(value - minimum) / (maximum - minimum) for value in processed]
