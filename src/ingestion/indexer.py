"""Builds and persists the two retrieval indexes: dense (Qdrant, on-disk, no
server required) and sparse (BM25, pickled to disk)."""
import pickle

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import embed_texts

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        if settings.qdrant_url:
            _qdrant_client = QdrantClient(url=settings.qdrant_url)
        else:
            _qdrant_client = QdrantClient(path=settings.qdrant_path)
    return _qdrant_client


def build_dense_index(chunks: list[Chunk]) -> None:
    client = get_qdrant_client()
    vectors = embed_texts([c.text for c in chunks])
    dim = len(vectors[0])

    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={"chunk_id": c.chunk_id, "text": c.text, "source": c.source, **c.metadata},
        )
        for i, c in enumerate(chunks)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    print(f"Indexed {len(points)} chunks into Qdrant collection '{settings.qdrant_collection}'")


def build_bm25_index(chunks: list[Chunk]) -> None:
    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(settings.bm25_index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"Persisted BM25 index with {len(chunks)} chunks to {settings.bm25_index_path}")


def build_all_indexes(chunks: list[Chunk]) -> None:
    build_dense_index(chunks)
    build_bm25_index(chunks)
