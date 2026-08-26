from typing import TypedDict

from src.retrieval.hybrid_search import RetrievedChunk


class AgentState(TypedDict, total=False):
    query: str
    original_query: str
    retrieved_chunks: list[RetrievedChunk]
    grade_verdict: str  # "sufficient" | "insufficient"
    grade_reason: str
    retry_count: int
    max_retries: int
    answer: str
    abstained: bool
    trace: list[dict]  # step-by-step log for observability/demo UI
