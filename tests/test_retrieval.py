from pathlib import Path

from app.models.entities import DocumentChunk
from app.services.embeddings import HashingEmbeddingProvider
from app.services.retrieval import HybridRetriever
from app.services.vector_store import LocalVectorStore


def _chunk(chunk_id: str, text: str, section: str = "General") -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        source="resume.pdf",
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={"section_name": section, "source_filename": "resume.pdf"},
    )


def test_hybrid_retriever_dedupes_and_returns_relevant_hits(tmp_path: Path) -> None:
    store = LocalVectorStore(index_dir=tmp_path / "index")
    embedder = HashingEmbeddingProvider(dimension=512)

    chunks = [
        _chunk("a1", "Skills: Python FastAPI Retrieval Augmented Generation", section="Skills"),
        _chunk("a2", "Skills: Python FastAPI Retrieval Augmented Generation", section="Skills"),
        _chunk("b1", "Experience: Built enterprise RAG assistant for recruiters", section="Experience"),
        _chunk("c1", "Education: Bachelor of Technology in Mining Engineering", section="Education"),
    ]
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
    store.upsert(chunks=chunks, embeddings=embeddings)

    retriever = HybridRetriever(vector_store=store)
    query_vector = embedder.embed_query("python fastapi skills")
    hits = retriever.retrieve(
        question="Show python fastapi skills",
        query_vector=query_vector,
        top_k=3,
        min_score=0.0,
    )

    assert hits
    assert len(hits) <= 3
    assert hits[0].chunk.metadata.get("section_name") in {"Skills", "Experience"}
    assert len({hit.chunk.text for hit in hits}) == len(hits)
