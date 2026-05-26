from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.core.config import Settings
from app.models.schemas import (
    IndexStatsResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ResetIndexResponse,
    SourceSnippet,
    UploadDocumentsResponse,
)
from app.services.rag_service import RagService
from app.services.text_cleaner import TextCleaner

router = APIRouter(prefix="/api/v1", tags=["RAG"])
text_cleaner = TextCleaner()


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


def get_runtime_settings(request: Request) -> Settings:
    return request.app.state.settings


def _resolve_source_dir(path_input: str | None, settings: Settings) -> Path:
    if path_input and path_input.strip():
        return Path(path_input.strip())
    return settings.raw_docs_dir


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    payload: IngestRequest,
    rag_service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_runtime_settings),
) -> IngestResponse:
    source_dir = _resolve_source_dir(payload.source_dir, settings)

    try:
        report = rag_service.ingest_directory(source_dir=source_dir, recursive=payload.recursive)
        return IngestResponse(**report)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@router.post("/upload", response_model=UploadDocumentsResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    source_dir: str | None = Form(default=None),
    recursive: bool = Form(default=True),
    ingest_after_upload: bool = Form(default=True),
    rag_service: RagService = Depends(get_rag_service),
    settings: Settings = Depends(get_runtime_settings),
) -> UploadDocumentsResponse:
    target_dir = _resolve_source_dir(source_dir, settings).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    allowed_extensions = set(settings.allowed_extensions)

    saved_files: list[str] = []
    skipped_files: list[str] = []

    for upload_file in files:
        filename = Path(upload_file.filename or "").name
        if not filename:
            skipped_files.append("(empty filename)")
            continue

        extension = Path(filename).suffix.lower()
        if extension not in allowed_extensions:
            skipped_files.append(filename)
            continue

        destination = target_dir / filename
        content = await upload_file.read()
        destination.write_bytes(content)
        saved_files.append(str(destination))

    if not saved_files:
        raise HTTPException(
            status_code=400,
            detail=(
                "No supported files uploaded. Allowed types: "
                f"{', '.join(sorted(allowed_extensions))}."
            ),
        )

    ingest_report = None
    if ingest_after_upload:
        try:
            report = rag_service.ingest_directory(source_dir=target_dir, recursive=recursive)
            ingest_report = IngestResponse(**report)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Ingestion failed after upload: {exc}") from exc

    return UploadDocumentsResponse(
        uploaded_files=len(saved_files),
        saved_to=str(target_dir),
        saved_files=saved_files,
        skipped_files=skipped_files,
        ingest_report=ingest_report,
    )


@router.post("/query", response_model=QueryResponse)
def ask_question(
    payload: QueryRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    answer, hits = rag_service.query(question=payload.question, top_k=payload.top_k)

    sources = [
        SourceSnippet(
            source=hit.chunk.source,
            filename=str(hit.chunk.metadata.get("source_filename") or Path(hit.chunk.source).name),
            section_name=str(hit.chunk.metadata.get("section_name", "General")),
            page_number=(
                int(hit.chunk.metadata["page_number"])
                if isinstance(hit.chunk.metadata.get("page_number"), int)
                else None
            ),
            score=round(hit.score, 4),
            vector_score=round(hit.vector_score, 4),
            lexical_score=round(hit.lexical_score, 4),
            excerpt=text_cleaner.normalize_excerpt(hit.chunk.text, max_chars=220),
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

