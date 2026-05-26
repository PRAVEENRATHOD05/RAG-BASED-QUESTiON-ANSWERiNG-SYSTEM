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
- Browser-based file upload + auto-ingest via React UI
- OCR/PDF text cleaning and unicode normalization
- Section-aware semantic chunking with overlap
- Chunk metadata enrichment: filename, section, page number
- Pluggable embedding backends
- Local persistent vector store
- Hybrid retrieval (vector + lexical) with reranking and deduplication
- Recruiter-ready grounded answer generation with confidence scoring
- Structured source attribution (`filename`, `section`, `page`)
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
frontend/
  src/
    App.jsx                 # React Q&A dashboard
    styles.css              # Modern responsive UI theme
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

### 8) Run the React UI

In a second terminal (keep backend running):

```bash
cd frontend
npm install
npm run dev
```

Open:

`http://127.0.0.1:5173`

Optional: create `frontend/.env` from `frontend/.env.example` and set `VITE_API_BASE_URL` if your API is not on `http://127.0.0.1:8000`.

The UI includes:
- Query panel with answer + source snippets
- Upload + ingest controls
- Index stats + reset actions
- `Source Directory` expects a folder path (not a terminal command)

After upgrading ingestion/chunking logic, run `DELETE /api/v1/index` once and ingest again so new metadata (`section_name`, `page_number`) is applied to all chunks.

---

## API Reference

- `GET /api/v1/health`
  - Service health check
- `POST /api/v1/ingest`
  - Ingest and index documents
- `POST /api/v1/upload`
  - Upload files (`multipart/form-data`) and optionally ingest immediately
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

Query responses include:
- structured answer block (`Answer`, `Confidence`, `Sources`)
- cleaned source snippets
- source metadata: `filename`, `section_name`, `page_number`, `score`, `vector_score`, `lexical_score`

### Upload and ingest a local file

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/upload" ^
  -F "files=@data/raw/sample_company_handbook.md" ^
  -F "ingest_after_upload=true"
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
- `SCORE_THRESHOLD` - minimum blended relevance score for retrieval
- `EMBEDDING_BACKEND` - `hashing` or `sentence-transformers`
- `HYBRID_VECTOR_WEIGHT` - vector relevance weight in hybrid retrieval
- `HYBRID_LEXICAL_WEIGHT` - lexical (BM25-style) relevance weight in hybrid retrieval
- `CORS_ALLOW_ORIGINS` - comma-separated frontend origins (for React dev server, etc.)
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

