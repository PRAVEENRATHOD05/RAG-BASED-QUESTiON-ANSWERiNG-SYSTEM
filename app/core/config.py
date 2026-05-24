from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "RAG-Based Question Answering System"
    project_description: str = (
        "A document-based Q&A service built using a Retrieval-Augmented "
        "Generation workflow."
    )
    api_version: str = "1.0.0"
    debug: bool = False

    data_dir: Path = Path("data")
    raw_docs_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    index_dir: Path = Path("data/vector_store")

    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx")
    chunk_size: int = 750
    chunk_overlap: int = 140

    top_k: int = 4
    score_threshold: float = 0.18

    embedding_backend: str = "hashing"
    embedding_dimension: int = 1536

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4.1-mini"
    allow_openai_generation: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_flag(cls, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False

        normalized = str(value).strip().lower()
        truthy = {"1", "true", "yes", "on", "debug"}
        falsy = {"0", "false", "no", "off", "release", "prod", "production"}

        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        return False

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_docs_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
