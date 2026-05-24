from __future__ import annotations

import logging
from pathlib import Path

from app.models.entities import LoadedDocument

logger = logging.getLogger(__name__)


class DocumentLoader:
    def __init__(self, allowed_extensions: tuple[str, ...]) -> None:
        self.allowed_extensions = {ext.lower() for ext in allowed_extensions}

    def discover_files(self, source_dir: Path, recursive: bool = True) -> list[Path]:
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        if not source_dir.is_dir():
            raise NotADirectoryError(f"Expected a directory path: {source_dir}")

        iterator = source_dir.rglob("*") if recursive else source_dir.glob("*")
        files = [
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in self.allowed_extensions
        ]
        return sorted(files)

    def load_file(self, file_path: Path) -> LoadedDocument:
        extension = file_path.suffix.lower()

        if extension == ".txt" or extension == ".md":
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        elif extension == ".pdf":
            text = self._read_pdf(file_path)
        elif extension == ".docx":
            text = self._read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        cleaned = text.replace("\x00", " ").strip()
        if not cleaned:
            raise ValueError(f"No readable text found in {file_path.name}")

        return LoadedDocument(
            source=str(file_path),
            text=cleaned,
            metadata={
                "extension": extension,
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
            },
        )

    def _read_pdf(self, file_path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF support requires 'pypdf'. Install dependencies from requirements.txt."
            ) from exc

        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def _read_docx(self, file_path: Path) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError(
                "DOCX support requires 'python-docx'. Install dependencies from requirements.txt."
            ) from exc

        doc = Document(str(file_path))
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(paragraphs)

