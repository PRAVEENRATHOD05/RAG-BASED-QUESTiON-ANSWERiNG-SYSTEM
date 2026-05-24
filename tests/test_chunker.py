from app.models.entities import LoadedDocument
from app.services.chunker import TextChunker


def test_chunker_splits_and_preserves_overlap() -> None:
    long_text = (
        "RAG systems combine retrieval and generation. "
        "Chunking helps make retrieval efficient and scalable. "
        "A good overlap keeps context continuity between chunks. "
    ) * 12
    document = LoadedDocument(source="notes.md", text=long_text)

    chunker = TextChunker(chunk_size=220, chunk_overlap=50)
    chunks = chunker.split_document(document)

    assert len(chunks) > 1
    assert chunks[0].start_char == 0
    assert chunks[1].start_char < chunks[0].end_char
    assert len({chunk.id for chunk in chunks}) == len(chunks)

