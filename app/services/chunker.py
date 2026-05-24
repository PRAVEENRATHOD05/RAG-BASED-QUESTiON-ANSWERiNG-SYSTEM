from __future__ import annotations

import hashlib
import re
from typing import Iterable

from app.models.entities import DocumentChunk, LoadedDocument


class TextChunker:
    def __init__(self, chunk_size: int = 750, chunk_overlap: int = 140) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: Iterable[LoadedDocument]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def split_document(self, document: LoadedDocument) -> list[DocumentChunk]:
        normalized_text = self._normalize(document.text)
        if not normalized_text:
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        text_length = len(normalized_text)
        min_break = int(self.chunk_size * 0.55)

        while start < text_length:
            tentative_end = min(start + self.chunk_size, text_length)
            end = self._find_natural_break(normalized_text, start, tentative_end, min_break)
            segment = normalized_text[start:end].strip()

            if segment:
                chunk_id = self._build_chunk_id(document.source, start, end, segment)
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        source=document.source,
                        text=segment,
                        start_char=start,
                        end_char=end,
                        metadata=document.metadata.copy(),
                    )
                )

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, 0)
            if start >= end:
                start = end

        return chunks

    def _normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _find_natural_break(
        self,
        text: str,
        start: int,
        end: int,
        min_break: int,
    ) -> int:
        if end >= len(text):
            return end

        window = text[start:end]
        breakpoints = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        absolute_min = start + min_break

        for token in breakpoints:
            index = window.rfind(token)
            if index == -1:
                continue
            candidate = start + index + len(token)
            if candidate >= absolute_min:
                return candidate

        return end

    def _build_chunk_id(self, source: str, start: int, end: int, text: str) -> str:
        signature = f"{source}|{start}|{end}|{text[:80]}"
        return hashlib.sha1(signature.encode("utf-8")).hexdigest()

