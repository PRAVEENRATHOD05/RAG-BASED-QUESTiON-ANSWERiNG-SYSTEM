from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}")


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbeddingProvider:
    """
    Lightweight local embedding model based on hashed token vectors.

    It is fast, deterministic, and has no heavy model dependency.
    """

    def __init__(self, dimension: int = 1536) -> None:
        if dimension < 128:
            raise ValueError("Embedding dimension must be at least 128")
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = [self._vectorize(text) for text in texts]
        matrix = np.vstack(vectors).astype(np.float32)
        return self._normalize_rows(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._vectorize(text).astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def _vectorize(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for index, token in enumerate(tokens):
            weight = 1.0 / (1.0 + (index * 0.01))
            self._add_token(vector, token, weight)

            if index + 1 < len(tokens):
                bigram = f"{token}_{tokens[index + 1]}"
                self._add_token(vector, bigram, weight * 0.65)

        return vector

    def _tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _add_token(self, vector: np.ndarray, token: str, weight: float) -> None:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        position = value % self.dimension
        sign = -1.0 if (value >> 1) & 1 else 1.0
        vector[position] += sign * weight

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install 'sentence-transformers' to use this embedding backend."
            ) from exc

        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(embedding, dtype=np.float32)


def create_embedding_provider(backend: str, dimension: int) -> EmbeddingProvider:
    backend = backend.lower().strip()
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbeddingProvider()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not initialize sentence-transformers backend (%s). "
                "Falling back to hashing embeddings.",
                exc,
            )

    return HashingEmbeddingProvider(dimension=dimension)

