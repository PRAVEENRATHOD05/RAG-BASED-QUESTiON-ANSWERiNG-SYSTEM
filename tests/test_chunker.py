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


def test_chunker_preserves_section_and_page_metadata() -> None:
    doc_text = (
        "SUMMARY\n"
        "AI engineer building retrieval systems.\n\n"
        "SKILLS\n"
        "Python, FastAPI, Retrieval, Vector Search.\n\n"
        "PROJECTS\n"
        "Built a resume assistant with confidence scoring and source tracing."
    )
    document = LoadedDocument(
        source="resume.pdf",
        text=doc_text,
        metadata={
            "filename": "resume.pdf",
            "page_map": [
                {"page_number": 1, "start_char": 0, "end_char": len(doc_text)},
            ],
        },
    )

    chunker = TextChunker(chunk_size=90, chunk_overlap=20)
    chunks = chunker.split_document(document)

    assert chunks
    assert all("section_name" in chunk.metadata for chunk in chunks)
    assert all(chunk.metadata.get("page_number") == 1 for chunk in chunks)
    assert any(chunk.metadata.get("section_name") == "Skills" for chunk in chunks)

