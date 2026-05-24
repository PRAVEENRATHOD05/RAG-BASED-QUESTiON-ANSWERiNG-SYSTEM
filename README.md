# RAG-Based Question Answering System

A production-style Retrieval-Augmented Generation (RAG) backend built with Python and FastAPI.

This project ingests local documents, splits them into meaningful chunks, creates vector embeddings, retrieves relevant context for a user question, and returns a grounded answer through a REST API.

---

## Why This Project Exists

Most Q&A systems fail when they answer without context.  
This system solves that by retrieving relevant document chunks first, then generating an answer from those chunks.

In simple words:
1. You provide documents.
2. The system indexes them.
3. You ask a question.
4. The system answers using your data.

---

## Features

- Multi-format document ingestion: `.txt`, `.md`, `.pdf`, `.docx`
- Configurable chunking with overlap for context continuity
- Pluggable embedding backends
- Local persistent vector store
- Semantic retrieval with similarity scoring
- Grounded answer generation with source snippets
- REST API for ingestion, querying, stats, and index reset
- Unit and API smoke tests

---

## Tech Stack

- Python 3.13
- FastAPI + Uvicorn
- NumPy
- Pydantic Settings
- pypdf + python-docx
- Optional OpenAI SDK for model-based answer generation

---

## Folder Structure

```text
app/
  api/
    routes.py               # API endpoints
  core/
    config.py               # Env-based settings
    logging.py              # Logging config
  models/
    entities.py             # Internal entities
    schemas.py              # Request/response models
  services/
    answer_generator.py     # Answer synthesis layer
    chunker.py              # Text chunking logic
    document_loader.py      # Document parsers
    embeddings.py           # Embedding providers
    rag_service.py          # Main orchestration service
    vector_store.py         # Persistent vector index
  main.py
scripts/
  ingest_docs.py            # CLI ingestion utility
tests/
data/
  raw/                      # Put source docs here
  processed/                # Ingestion manifests
  vector_store/             # Saved index files
```

---

## Quick Start (Beginner Friendly)

### 1) Create virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Create environment file

```bash
copy .env.example .env
```

### 3) Add your docs

Drop your files into:

`data/raw`

### 4) Start API server

```bash
uvicorn app.main:app --reload
```

### 5) Open Swagger docs

`http://127.0.0.1:8000/docs`

### 6) Ingest documents

Use endpoint: `POST /api/v1/ingest`

Sample payload:

```json
{}
```

### 7) Ask questions

Use endpoint: `POST /api/v1/query`

Sample payload:

```json
{
  "question": "How many remote work days are allowed per week?"
}
```

---

## API Reference

- `GET /api/v1/health`
  - Service health check
- `POST /api/v1/ingest`
  - Ingest and index documents
- `POST /api/v1/query`
  - Ask a question against indexed documents
- `GET /api/v1/index/stats`
  - View index metadata (chunks, dimensions, sources)
- `DELETE /api/v1/index`
  - Reset/clear the index

---

## cURL Examples (Windows)

### Ingest documents from default `data/raw`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/ingest" ^
  -H "Content-Type: application/json" ^
  -d "{}"
```

### Query the system

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"How many remote days are allowed each week?\"}"
```

---

## CLI Utility

You can ingest without API using:

```bash
python scripts/ingest_docs.py --source-dir data/raw
```

---

## Configuration

Main knobs in `.env`:

- `CHUNK_SIZE` - characters per chunk
- `CHUNK_OVERLAP` - overlap between consecutive chunks
- `TOP_K` - number of retrieved chunks
- `SCORE_THRESHOLD` - minimum similarity score
- `EMBEDDING_BACKEND` - `hashing` or `sentence-transformers`
- `ALLOW_OPENAI_GENERATION` - `true/false`
- `OPENAI_API_KEY` and `OPENAI_MODEL` (if OpenAI generation is enabled)

---

## Embedding Backends

### Default: `hashing`

- Fast
- Deterministic
- No external model download
- Good for local development and demos

### Optional: `sentence-transformers`

- Better semantic quality
- Requires extra package/model download

Set:

```env
EMBEDDING_BACKEND=sentence-transformers
```

---

## Testing

Run full tests:

```bash
pytest -q
```

Current coverage includes:
- chunking behavior
- vector store upsert and retrieval flow
- API ingest + query smoke path

---

## Architecture Notes (5 Years Exp View)

- Clear service boundaries keep business logic independent from HTTP concerns.
- `RagService` acts as orchestration point for ingestion and query pipelines.
- Vector index is persisted locally (`chunks.json` + `embeddings.npy`) for repeatable runs.
- Answer generation supports both local extractive mode and optional LLM-based mode.
- Configuration is centralized and environment-driven, which keeps deployment flexible.

---

## Future Improvements

- Add reranking layer for better retrieval precision
- Add hybrid search (keyword + vector)
- Add async batch ingestion for very large corpora
- Add authentication and request-level usage limits
- Add Dockerfile + CI workflow for one-command deployment

