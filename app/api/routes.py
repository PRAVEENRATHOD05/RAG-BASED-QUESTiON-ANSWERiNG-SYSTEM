from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings, get_settings
from app.models.schemas import (
    IndexStatsResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ResetIndexResponse,
    SourceSnippet,
)
from app.services.rag_service import RagService

router = APIRouter(prefix="/api/v1", tags=["RAG"])


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    payload: IngestRequest,
    rag_service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    source_dir = Path(payload.source_dir) if payload.source_dir else settings.raw_docs_dir

    try:
        report = rag_service.ingest_directory(source_dir=source_dir, recursive=payload.recursive)
        return IngestResponse(**report)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@router.post("/query", response_model=QueryResponse)
def ask_question(
    payload: QueryRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    answer, hits = rag_service.query(question=payload.question, top_k=payload.top_k)

    sources = [
        SourceSnippet(
            source=hit.chunk.source,
            score=round(hit.score, 4),
            excerpt=hit.chunk.text[:350],
            start_char=hit.chunk.start_char,
            end_char=hit.chunk.end_char,
        )
        for hit in hits
    ]
    return QueryResponse(
        question=payload.question,
        answer=answer,
        retrieved_chunks=len(hits),
        sources=sources,
    )


@router.get("/index/stats", response_model=IndexStatsResponse)
def get_index_stats(
    rag_service: RagService = Depends(get_rag_service),
) -> IndexStatsResponse:
    return IndexStatsResponse(**rag_service.stats())


@router.delete("/index", response_model=ResetIndexResponse)
def reset_index(
    rag_service: RagService = Depends(get_rag_service),
) -> ResetIndexResponse:
    removed = rag_service.reset_index()
    return ResetIndexResponse(removed_chunks=removed)

