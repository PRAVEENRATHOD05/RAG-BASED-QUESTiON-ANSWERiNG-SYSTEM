from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LoadedDocument:
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    id: str
    source: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentChunk":
        return cls(
            id=payload["id"],
            source=payload["source"],
            text=payload["text"],
            start_char=payload["start_char"],
            end_char=payload["end_char"],
            metadata=payload.get("metadata", {}),
        )


@dataclass(slots=True)
class SearchHit:
    chunk: DocumentChunk
    score: float

