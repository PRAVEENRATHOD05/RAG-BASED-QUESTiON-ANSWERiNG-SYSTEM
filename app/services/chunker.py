from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from app.models.entities import DocumentChunk, LoadedDocument

SECTION_HINTS = {
    "summary",
    "profile",
    "objective",
    "experience",
    "work experience",
    "internship",
    "internships",
    "projects",
    "education",
    "skills",
    "technical skills",
    "achievements",
    "certifications",
    "contact",
}


class TextChunker:
    """Semantic-first chunking with section preservation and overlap."""

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

        sections = self._split_sections(normalized_text)
        chunks: list[DocumentChunk] = []
        seen_signatures: set[str] = set()
        chunk_order = 0

        for section_name, section_start, section_text in sections:
            local_windows = self._semantic_windows(section_text)
            for local_start, local_end in local_windows:
                global_start = section_start + local_start
                global_end = section_start + local_end
                segment = section_text[local_start:local_end].strip()
                clean_segment = self._compact(segment)

                if len(clean_segment) < 35 or self._is_noise(clean_segment):
                    continue

                dedupe_signature = clean_segment.lower()[:220]
                if dedupe_signature in seen_signatures:
                    continue
                seen_signatures.add(dedupe_signature)

                metadata = document.metadata.copy()
                metadata["section_name"] = section_name
                metadata["chunk_order"] = chunk_order
                metadata["page_number"] = self._resolve_page_number(
                    global_start,
                    global_end,
                    metadata.get("page_map"),
                )

                chunk_id = self._build_chunk_id(
                    source=document.source,
                    section_name=section_name,
                    start=global_start,
                    end=global_end,
                    text=clean_segment,
                )
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        source=document.source,
                        text=clean_segment,
                        start_char=global_start,
                        end_char=global_end,
                        metadata=metadata,
                    )
                )
                chunk_order += 1

        return chunks

    def _normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_sections(self, text: str) -> list[tuple[str, int, str]]:
        heading_positions = self._detect_heading_positions(text)
        if not heading_positions:
            return [("General", 0, text)]

        sections: list[tuple[str, int, str]] = []
        cursor = 0

        first_heading_start = heading_positions[0][1]
        if first_heading_start > 20:
            intro = text[:first_heading_start].strip()
            if intro:
                sections.append(("General", 0, intro))

        for index, (heading_name, heading_start, heading_end) in enumerate(heading_positions):
            next_start = heading_positions[index + 1][1] if index + 1 < len(heading_positions) else len(text)
            body_text = text[heading_end:next_start].strip()
            if not body_text:
                continue

            body_start = text.find(body_text, heading_end)
            if body_start < 0:
                body_start = cursor

            sections.append((heading_name, body_start, body_text))
            cursor = next_start

        if not sections:
            return [("General", 0, text)]

        return sections

    def _detect_heading_positions(self, text: str) -> list[tuple[str, int, int]]:
        positions: list[tuple[str, int, int]] = []
        line_pattern = re.compile(r"(?m)^.*$")

        for match in line_pattern.finditer(text):
            line = match.group().strip()
            if not line:
                continue
            if self._is_heading(line):
                section_name = self._normalize_heading(line)
                positions.append((section_name, match.start(), match.end()))

        unique: list[tuple[str, int, int]] = []
        seen_starts: set[int] = set()
        for item in positions:
            if item[1] in seen_starts:
                continue
            seen_starts.add(item[1])
            unique.append(item)
        return unique

    def _is_heading(self, line: str) -> bool:
        cleaned = line.strip(" :-|#•\t")
        if len(cleaned) < 3 or len(cleaned) > 70:
            return False

        if "," in cleaned and cleaned.count(",") >= 1 and len(cleaned) > 18:
            return False
        if "." in cleaned and not cleaned.endswith("."):
            return False

        lowered = cleaned.lower()
        if lowered in SECTION_HINTS:
            return True
        if lowered.endswith(" experience") or lowered.endswith(" skills"):
            return True
        if line.startswith("#"):
            return True
        if cleaned.endswith(":"):
            return True

        alpha_chars = [char for char in cleaned if char.isalpha()]
        if not alpha_chars:
            return False
        uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
        if uppercase_ratio > 0.72:
            return True

        words = cleaned.split()
        if len(words) <= 6 and all(word[:1].isupper() for word in words if word):
            return True
        return False

    def _normalize_heading(self, line: str) -> str:
        heading = line.strip(" :-|#•\t")
        heading = re.sub(r"\s+", " ", heading).strip()
        return heading.title() if heading else "General"

    def _semantic_windows(self, text: str) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        text_length = len(text)
        if text_length <= self.chunk_size:
            return [(0, text_length)]

        start = 0
        min_break = int(self.chunk_size * 0.58)
        while start < text_length:
            tentative_end = min(start + self.chunk_size, text_length)
            end = self._find_natural_break(text, start, tentative_end, min_break)

            if end <= start:
                end = min(start + self.chunk_size, text_length)
            windows.append((start, end))
            if end >= text_length:
                break

            next_start = max(end - self.chunk_overlap, 0)
            if next_start <= start:
                next_start = end
            start = next_start

        return windows

    def _find_natural_break(self, text: str, start: int, end: int, min_break: int) -> int:
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

    def _resolve_page_number(self, start: int, end: int, page_map: object) -> int | None:
        if not isinstance(page_map, list) or not page_map:
            return None

        best_page: int | None = None
        best_overlap = -1

        for entry in page_map:
            if not isinstance(entry, dict):
                continue
            page_start = int(entry.get("start_char", 0))
            page_end = int(entry.get("end_char", 0))
            page_number = int(entry.get("page_number", 0))
            overlap = min(end, page_end) - max(start, page_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_page = page_number

        return best_page if best_page and best_overlap > 0 else None

    def _is_noise(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        alpha_ratio = sum(1 for char in stripped if char.isalpha()) / max(len(stripped), 1)
        return alpha_ratio < 0.35

    def _compact(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text

    def _build_chunk_id(self, source: str, section_name: str, start: int, end: int, text: str) -> str:
        signature = f"{source}|{section_name}|{start}|{end}|{text[:120]}"
        return hashlib.sha1(signature.encode("utf-8")).hexdigest()
