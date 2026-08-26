"""FastAPI demo surface: shows the full retrieve/grade/correct/answer trace
per query, so the self-correction behavior is visible, not just the final answer."""
from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.graph import ask

app = FastAPI(title="Adaptive Retrieval Agent")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    retries_used: int
    sources: list[str]
    trace: list[dict]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = ask(request.question)
    return QueryResponse(
        answer=result["answer"],
        abstained=result["abstained"],
        retries_used=result["retry_count"],
        sources=sorted({c.source for c in result.get("retrieved_chunks", [])}),
        trace=result.get("trace", []),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
