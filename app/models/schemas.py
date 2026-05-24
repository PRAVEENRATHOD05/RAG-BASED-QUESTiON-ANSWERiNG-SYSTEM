from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_dir: str | None = Field(
        default=None,
        description="Path to source documents folder. If omitted, defaults to data/raw.",
    )
    recursive: bool = Field(default=True)


class IngestResponse(BaseModel):
    files_discovered: int
    files_processed: int
    chunks_created: int
    chunks_added: int
    chunks_updated: int
    skipped_files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceSnippet(BaseModel):
    source: str
    score: float
    excerpt: str
    start_char: int
    end_char: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: int
    sources: list[SourceSnippet]


class IndexStatsResponse(BaseModel):
    total_chunks: int
    embedding_dimension: int
    indexed_sources: int


class ResetIndexResponse(BaseModel):
    removed_chunks: int

