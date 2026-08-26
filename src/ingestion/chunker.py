"""Chunks markdown documents into overlapping passages, keeping source metadata
attached to every chunk so answers can cite back to it."""
import glob
import os
from dataclasses import dataclass, field

from langchain_text_splitters import MarkdownTextSplitter


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def load_and_chunk(raw_docs_path: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Chunk] = []

    for file_path in sorted(glob.glob(os.path.join(raw_docs_path, "*.md"))):
        source_name = os.path.basename(file_path)
        with open(file_path, encoding="utf-8") as f:
            text = f.read()

        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            chunk_id = f"{source_name}::chunk{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    source=source_name,
                    metadata={"chunk_index": i, "source": source_name},
                )
            )

    return chunks


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.config import settings

    result = load_and_chunk(settings.raw_docs_path)
    print(f"Produced {len(result)} chunks from {settings.raw_docs_path}")
    if result:
        print("Example chunk:", result[0].chunk_id, "-", result[0].text[:120].replace("\n", " "), "...")
