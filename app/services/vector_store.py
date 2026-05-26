from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.models.entities import DocumentChunk, SearchHit


class LocalVectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.chunks_file = index_dir / "chunks.json"
        self.embeddings_file = index_dir / "embeddings.npy"

        self._loaded = False
        self._chunks: list[DocumentChunk] = []
        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._id_to_index: dict[str, int] = {}

    def load(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if self.chunks_file.exists() and self.embeddings_file.exists():
            chunks_payload = json.loads(self.chunks_file.read_text(encoding="utf-8"))
            chunks = [DocumentChunk.from_dict(item) for item in chunks_payload]
            embeddings = np.load(self.embeddings_file).astype(np.float32)

            if embeddings.ndim != 2:
                raise ValueError("Invalid embeddings file shape.")
            if len(chunks) != embeddings.shape[0]:
                raise ValueError("Mismatch between chunk records and embedding vectors.")

            self._chunks = chunks
            self._embeddings = embeddings
            self._id_to_index = {chunk.id: idx for idx, chunk in enumerate(chunks)}
        else:
            self._chunks = []
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            self._id_to_index = {}

        self._loaded = True

    def save(self) -> None:
        self._ensure_loaded()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = [chunk.to_dict() for chunk in self._chunks]
        self.chunks_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        np.save(self.embeddings_file, self._embeddings)

    def clear(self) -> int:
        self._ensure_loaded()
        removed = len(self._chunks)
        if self.chunks_file.exists():
            self.chunks_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        self._chunks = []
        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._id_to_index = {}
        return removed

    def upsert(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> tuple[int, int]:
        self._ensure_loaded()
        if not chunks:
            return (0, 0)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunk count and embedding count must match.")

        added = 0
        updated = 0

        if self._embeddings.size == 0:
            self._embeddings = np.zeros((0, embeddings.shape[1]), dtype=np.float32)
        elif self._embeddings.shape[1] != embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: store={self._embeddings.shape[1]}, "
                f"incoming={embeddings.shape[1]}"
            )

        new_chunks: list[DocumentChunk] = []
        new_vectors: list[np.ndarray] = []

        for chunk, vector in zip(chunks, embeddings, strict=True):
            if chunk.id in self._id_to_index:
                idx = self._id_to_index[chunk.id]
                self._chunks[idx] = chunk
                self._embeddings[idx] = vector
                updated += 1
            else:
                new_chunks.append(chunk)
                new_vectors.append(vector)
                added += 1

        if new_chunks:
            matrix = np.vstack(new_vectors).astype(np.float32)
            self._embeddings = np.vstack([self._embeddings, matrix])
            start_index = len(self._chunks)
            self._chunks.extend(new_chunks)
            for offset, chunk in enumerate(new_chunks):
                self._id_to_index[chunk.id] = start_index + offset

        self.save()
        return (added, updated)

    def search(self, query_vector: np.ndarray, top_k: int, min_score: float) -> list[SearchHit]:
        self._ensure_loaded()
        if self._embeddings.size == 0:
            return []

        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Query vector must be 1D.")
        if vector.shape[0] != self._embeddings.shape[1]:
            raise ValueError(
                f"Query dimension mismatch: got {vector.shape[0]}, "
                f"expected {self._embeddings.shape[1]}"
            )

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        scores = self._embeddings @ vector
        top_k = min(max(top_k, 1), len(scores))
        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        hits: list[SearchHit] = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            hits.append(
                SearchHit(
                    chunk=self._chunks[int(idx)],
                    score=score,
                    vector_score=score,
                    lexical_score=0.0,
                )
            )
        return hits

    def score_all(self, query_vector: np.ndarray) -> list[tuple[DocumentChunk, float]]:
        self._ensure_loaded()
        if self._embeddings.size == 0:
            return []

        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("Query vector must be 1D.")
        if vector.shape[0] != self._embeddings.shape[1]:
            raise ValueError(
                f"Query dimension mismatch: got {vector.shape[0]}, "
                f"expected {self._embeddings.shape[1]}"
            )

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        scores = self._embeddings @ vector
        ranking = np.argsort(scores)[::-1]
        return [(self._chunks[int(idx)], float(scores[idx])) for idx in ranking]

    def all_chunks(self) -> list[DocumentChunk]:
        self._ensure_loaded()
        return list(self._chunks)

    def stats(self) -> dict[str, int]:
        self._ensure_loaded()
        dimension = self._embeddings.shape[1] if self._embeddings.ndim == 2 and self._embeddings.size else 0
        source_count = len({chunk.source for chunk in self._chunks})
        return {
            "total_chunks": len(self._chunks),
            "embedding_dimension": int(dimension),
            "indexed_sources": source_count,
        }

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

