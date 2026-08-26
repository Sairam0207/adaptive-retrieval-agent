import os
import tempfile

from src.ingestion.chunker import load_and_chunk


def test_chunks_carry_source_metadata():
    with tempfile.TemporaryDirectory() as tmp_dir:
        doc_path = os.path.join(tmp_dir, "sample.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("# Title\n\n" + ("This is a sentence about FastAPI. " * 200))

        chunks = load_and_chunk(tmp_dir, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1
        assert all(c.source == "sample.md" for c in chunks)
        assert all(c.chunk_id.startswith("sample.md::chunk") for c in chunks)


def test_no_documents_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert load_and_chunk(tmp_dir) == []
