"""Cross-encoder reranking: scores query-chunk relevance more precisely than
the initial hybrid retrieval, narrowing candidates before the grading step."""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.hybrid_search import RetrievedChunk


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
    if not chunks:
        return []

    top_k = top_k or settings.top_k_rerank
    model = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs)

    reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(chunk_id=c.chunk_id, text=c.text, source=c.source, score=float(s))
        for c, s in reranked
    ]
