from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.models.entities import SearchHit
from app.services.answer_generator import build_answer_generator
from app.services.chunker import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.embeddings import create_embedding_provider
from app.services.retrieval import HybridRetriever
from app.services.text_cleaner import TextCleaner
from app.services.vector_store import LocalVectorStore


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cleaner = TextCleaner()
        self.loader = DocumentLoader(settings.allowed_extensions, cleaner=self.cleaner)
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embedding_provider = create_embedding_provider(
            backend=settings.embedding_backend,
            dimension=settings.embedding_dimension,
        )
        self.vector_store = LocalVectorStore(index_dir=settings.index_dir)
        self.vector_store.load()
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            vector_weight=settings.hybrid_vector_weight,
            lexical_weight=settings.hybrid_lexical_weight,
        )
        self.answer_generator = build_answer_generator(
            use_openai=settings.allow_openai_generation,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )

    def ingest_directory(self, source_dir: Path, recursive: bool = True) -> dict[str, Any]:
        source_dir = source_dir.resolve()
        files = self.loader.discover_files(source_dir=source_dir, recursive=recursive)

        report: dict[str, Any] = {
            "files_discovered": len(files),
            "files_processed": 0,
            "chunks_created": 0,
            "chunks_added": 0,
            "chunks_updated": 0,
            "skipped_files": [],
            "errors": [],
        }

        documents = []
        for file_path in files:
            try:
                document = self.loader.load_file(file_path)
                documents.append(document)
                report["files_processed"] += 1
            except Exception as exc:  # noqa: BLE001
                report["skipped_files"].append(str(file_path))
                report["errors"].append(f"{file_path.name}: {exc}")

        chunks = self.chunker.split_documents(documents)
        report["chunks_created"] = len(chunks)

        if chunks:
            embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
            added, updated = self.vector_store.upsert(chunks, embeddings)
            report["chunks_added"] = added
            report["chunks_updated"] = updated

        self._write_ingestion_manifest(source_dir=source_dir, report=report)
        return report

    def query(self, question: str, top_k: int | None = None) -> tuple[str, list[SearchHit]]:
        question = self.cleaner.clean(question)
        if not question:
            return (
                "Answer:\nI could not find that information in the uploaded document.\n\n"
                "Confidence:\nLow\n\n"
                "Sources:\n- Not available",
                [],
            )

        limit = top_k or self.settings.top_k
        limit = max(1, min(limit, 20))
        query_vector = self.embedding_provider.embed_query(question)
        hits = self.retriever.retrieve(
            question=question,
            query_vector=query_vector,
            top_k=limit,
            min_score=self.settings.score_threshold,
        )
        if not hits:
            hits = self.retriever.retrieve(
                question=question,
                query_vector=query_vector,
                top_k=limit,
                min_score=0.0,
            )
        answer = self.answer_generator.generate(question=question, hits=hits)
        return answer, hits

    def reset_index(self) -> int:
        return self.vector_store.clear()

    def stats(self) -> dict[str, int]:
        return self.vector_store.stats()

    def _write_ingestion_manifest(self, source_dir: Path, report: dict[str, Any]) -> None:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        manifest_path = self.settings.processed_dir / f"ingestion_{timestamp}.json"
        payload = {
            "timestamp_utc": timestamp,
            "source_dir": str(source_dir),
            "settings": {
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "embedding_backend": self.settings.embedding_backend,
            },
            "report": report,
        }
        manifest_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

