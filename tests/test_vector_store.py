from pathlib import Path

from app.models.entities import DocumentChunk
from app.services.embeddings import HashingEmbeddingProvider
from app.services.vector_store import LocalVectorStore


def make_chunk(chunk_id: str, source: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        source=source,
        text=text,
        start_char=0,
        end_char=len(text),
        metadata={},
    )


def test_vector_store_upsert_and_search(tmp_path: Path) -> None:
    store = LocalVectorStore(index_dir=tmp_path / "index")
    embedder = HashingEmbeddingProvider(dimension=512)

    chunks = [
        make_chunk("1", "policy.md", "Refunds are processed within 7 business days."),
        make_chunk("2", "support.md", "The support team is available 24/7 by email."),
        make_chunk("3", "shipping.md", "Standard shipping takes 3 to 5 working days."),
    ]
    embeddings = embedder.embed_texts([chunk.text for chunk in chunks])

    added, updated = store.upsert(chunks=chunks, embeddings=embeddings)
    assert added == 3
    assert updated == 0

    query_vector = embedder.embed_query("How long does a refund take?")
    hits = store.search(query_vector=query_vector, top_k=2, min_score=-1.0)

    assert len(hits) == 2
    assert hits[0].chunk.source in {"policy.md", "shipping.md"}

