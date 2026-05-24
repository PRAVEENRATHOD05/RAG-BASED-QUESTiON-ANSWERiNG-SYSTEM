from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.services.rag_service import RagService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest local documents into the RAG index.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing source files (txt, md, pdf, docx).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only ingest files directly under source-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    rag_service = RagService(settings=settings)

    source_dir = args.source_dir or settings.raw_docs_dir
    report = rag_service.ingest_directory(
        source_dir=source_dir,
        recursive=not args.non_recursive,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
