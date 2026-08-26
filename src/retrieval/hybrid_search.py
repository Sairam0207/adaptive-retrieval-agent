"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) search, merged with
Reciprocal Rank Fusion so exact-term matches and semantic matches both surface."""
import pickle
from dataclasses import dataclass

from src.config import settings
from src.ingestion.embedder import embed_query
from src.ingestion.indexer import get_qdrant_client

RRF_K = 60  # standard smoothing constant for reciprocal rank fusion


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    score: float


def _dense_search(query: str, top_k: int) -> list[RetrievedChunk]:
    client = get_qdrant_client()
    query_vector = embed_query(query)
    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
    ).points
    return [
        RetrievedChunk(
            chunk_id=hit.payload["chunk_id"],
            text=hit.payload["text"],
            source=hit.payload["source"],
            score=hit.score,
        )
        for hit in hits
    ]


def _sparse_search(query: str, top_k: int) -> list[RetrievedChunk]:
    with open(settings.bm25_index_path, "rb") as f:
        data = pickle.load(f)
    bm25 = data["bm25"]
    chunks = data["chunks"]

    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(chunk_id=c.chunk_id, text=c.text, source=c.source, score=float(s))
        for c, s in ranked
        if s > 0
    ]


def hybrid_search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k_retrieve

    dense_results = _dense_search(query, top_k)
    sparse_results = _sparse_search(query, top_k)

    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievedChunk] = {}

    for rank, result in enumerate(dense_results):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_lookup[result.chunk_id] = result

    for rank, result in enumerate(sparse_results):
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_lookup.setdefault(result.chunk_id, result)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=chunk_lookup[chunk_id].text,
            source=chunk_lookup[chunk_id].source,
            score=fused_score,
        )
        for chunk_id, fused_score in fused
    ]
