"""End-to-end ingestion CLI: fetch corpus (if empty) -> chunk -> embed -> index."""
import glob
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config import settings
from src.ingestion.chunker import load_and_chunk
from src.ingestion.fetch_corpus import fetch_all
from src.ingestion.indexer import build_all_indexes


def main() -> None:
    os.makedirs(settings.raw_docs_path, exist_ok=True)
    existing = glob.glob(os.path.join(settings.raw_docs_path, "*.md"))
    if not existing:
        print("No documents found in raw docs path, fetching sample corpus (FastAPI docs)...")
        fetch_all()

    chunks = load_and_chunk(settings.raw_docs_path)
    if not chunks:
        raise SystemExit(f"No chunks produced from {settings.raw_docs_path} — check the corpus.")

    print(f"Chunked {len(glob.glob(os.path.join(settings.raw_docs_path, '*.md')))} documents "
          f"into {len(chunks)} chunks.")
    build_all_indexes(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
