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
        assert "Answer:" in query_payload["answer"]
        assert "Confidence:" in query_payload["answer"]
        assert "Sources:" in query_payload["answer"]
        first_source = query_payload["sources"][0]
        assert "filename" in first_source
        assert "section_name" in first_source
        assert "vector_score" in first_source
        assert "lexical_score" in first_source


def test_upload_and_ingest_flow(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        raw_docs_dir=tmp_path / "data" / "raw",
        processed_dir=tmp_path / "data" / "processed",
        index_dir=tmp_path / "data" / "vector_store",
        score_threshold=0.95,
    )
    settings.ensure_directories()

    app = create_app(custom_settings=settings)
    with TestClient(app) as client:
        upload_response = client.post(
            "/api/v1/upload",
            data={"ingest_after_upload": "true", "recursive": "true"},
            files={
                "files": (
                    "remote_policy.md",
                    (
                        "Remote Work Policy\n\nEmployees can work remotely "
                        "up to three days per week."
                    ).encode("utf-8"),
                    "text/markdown",
                )
            },
        )
        assert upload_response.status_code == 200
        upload_payload = upload_response.json()
        assert upload_payload["uploaded_files"] == 1
        assert upload_payload["ingest_report"]["files_processed"] == 1
        stats_response = client.get("/api/v1/index/stats")
        assert stats_response.status_code == 200
        stats_payload = stats_response.json()
        assert stats_payload["total_chunks"] >= 1

        query_response = client.post("/api/v1/query", json={"question": "tell about remote policy"})
        assert query_response.status_code == 200
        query_payload = query_response.json()
        assert query_payload["retrieved_chunks"] >= 1

