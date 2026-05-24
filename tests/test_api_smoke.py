from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_ingest_and_query_flow(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        raw_docs_dir=tmp_path / "data" / "raw",
        processed_dir=tmp_path / "data" / "processed",
        index_dir=tmp_path / "data" / "vector_store",
        score_threshold=0.0,
    )
    settings.ensure_directories()

    sample_doc = settings.raw_docs_dir / "company_policy.md"
    sample_doc.write_text(
        (
            "Remote Work Policy\n\n"
            "Employees can work remotely up to three days per week with manager approval. "
            "Core collaboration hours are 11:00 AM to 4:00 PM."
        ),
        encoding="utf-8",
    )

    app = create_app(custom_settings=settings)
    with TestClient(app) as client:
        ingest_response = client.post("/api/v1/ingest", json={})
        assert ingest_response.status_code == 200
        ingest_payload = ingest_response.json()
        assert ingest_payload["files_processed"] == 1
        assert ingest_payload["chunks_created"] >= 1

        query_response = client.post(
            "/api/v1/query",
            json={"question": "How many days can employees work remotely?"},
        )
        assert query_response.status_code == 200
        query_payload = query_response.json()
        assert query_payload["retrieved_chunks"] >= 1
        assert query_payload["answer"]

