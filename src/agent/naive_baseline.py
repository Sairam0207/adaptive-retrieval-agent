"""Plain retrieve-then-generate pipeline with no grading or self-correction.
Exists solely as the comparison point for the eval harness — this is what
the corrective loop is measured against."""
from google.genai import types

from src.agent.llm_client import generate
from src.agent.nodes import GENERATE_SYSTEM_PROMPT
from src.config import settings
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank


def naive_answer(question: str) -> dict:
    hits = hybrid_search(question, top_k=settings.top_k_retrieve)
    top_chunks = rerank(question, hits, top_k=settings.top_k_rerank)
    context = "\n\n".join(f"[{c.source}] {c.text}" for c in top_chunks)

    response = generate(
        model=settings.generator_model,
        contents=f"Question: {question}\n\nContext:\n{context}",
        config=types.GenerateContentConfig(system_instruction=GENERATE_SYSTEM_PROMPT, max_output_tokens=800),
        fallback_model=settings.grader_model,
    )
    return {
        "answer": response.text.strip(),
        "sources": sorted({c.source for c in top_chunks}),
    }
