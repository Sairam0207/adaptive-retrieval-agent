"""The five nodes implementing the retrieve -> grade -> correct -> generate
loop. Grading and reformulation use the cheap model; generation uses the
strong model — this split is the project's cost-optimization story."""
import json

from google.genai import types

from src.agent.llm_client import generate
from src.agent.state import AgentState
from src.config import settings
from src.observability.tracing import traced
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank

GRADE_SYSTEM_PROMPT = """You grade whether retrieved context is sufficient to fully and \
accurately answer a user's question. Respond with ONLY a JSON object, no other text:
{"verdict": "sufficient" | "insufficient", "reason": "<one short sentence>"}
Mark "insufficient" if the context is missing facts needed to answer, is only \
tangentially related, or would require guessing to fill gaps."""

REFORMULATE_SYSTEM_PROMPT = """A retrieval system failed to find sufficient context for a \
question. Given the original question and why retrieval fell short, rewrite the question \
as a better search query: more specific, using terminology likely to appear in technical \
documentation. Respond with ONLY the rewritten query text, no explanation."""

GENERATE_SYSTEM_PROMPT = """Answer the user's question using ONLY the provided context. \
Cite the source of every claim inline using the format [source_file]. If the context does \
not fully support an answer, say so explicitly rather than guessing."""


@traced("retrieve")
def retrieve_node(state: AgentState) -> AgentState:
    hits = hybrid_search(state["query"], top_k=settings.top_k_retrieve)
    reranked = rerank(state["query"], hits, top_k=settings.top_k_rerank)

    state.setdefault("trace", []).append({
        "step": "retrieve",
        "query": state["query"],
        "num_hits": len(reranked),
        "chunk_ids": [c.chunk_id for c in reranked],
    })
    state["retrieved_chunks"] = reranked
    return state


@traced("grade")
def grade_node(state: AgentState) -> AgentState:
    context = "\n\n".join(f"[{c.source}] {c.text}" for c in state["retrieved_chunks"])
    response = generate(
        model=settings.grader_model,
        contents=f"Question: {state['original_query']}\n\nContext:\n{context}",
        config=types.GenerateContentConfig(
            system_instruction=GRADE_SYSTEM_PROMPT,
            max_output_tokens=200,
            response_mime_type="application/json",
        ),
    )
    raw = response.text.strip()

    try:
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "insufficient")
        reason = parsed.get("reason", "")
    except (json.JSONDecodeError, IndexError):
        verdict, reason = "insufficient", f"unparseable grader output: {raw[:100]}"

    state["grade_verdict"] = verdict
    state["grade_reason"] = reason
    state.setdefault("trace", []).append({"step": "grade", "verdict": verdict, "reason": reason})
    return state


def route_after_grade(state: AgentState) -> str:
    if state["grade_verdict"] == "sufficient":
        return "generate"
    if state["retry_count"] >= state["max_retries"]:
        return "abstain"
    return "reformulate"


@traced("reformulate")
def reformulate_node(state: AgentState) -> AgentState:
    response = generate(
        model=settings.grader_model,
        contents=(
            f"Original question: {state['original_query']}\n"
            f"Previous query tried: {state['query']}\n"
            f"Why retrieval was insufficient: {state['grade_reason']}"
        ),
        config=types.GenerateContentConfig(system_instruction=REFORMULATE_SYSTEM_PROMPT, max_output_tokens=150),
    )
    new_query = response.text.strip()

    state["retry_count"] = state["retry_count"] + 1
    state.setdefault("trace", []).append({
        "step": "reformulate",
        "old_query": state["query"],
        "new_query": new_query,
        "retry_count": state["retry_count"],
    })
    state["query"] = new_query
    return state


@traced("generate")
def generate_node(state: AgentState) -> AgentState:
    context = "\n\n".join(f"[{c.source}] {c.text}" for c in state["retrieved_chunks"])
    response = generate(
        model=settings.generator_model,
        contents=f"Question: {state['original_query']}\n\nContext:\n{context}",
        config=types.GenerateContentConfig(system_instruction=GENERATE_SYSTEM_PROMPT, max_output_tokens=800),
        fallback_model=settings.grader_model,
    )
    answer = response.text.strip()

    state["answer"] = answer
    state["abstained"] = False
    state.setdefault("trace", []).append({"step": "generate", "answer": answer})
    return state


@traced("abstain")
def abstain_node(state: AgentState) -> AgentState:
    state["answer"] = (
        "I don't have enough information in the knowledge base to answer this "
        f"question confidently. Last retrieval issue: {state.get('grade_reason', 'unknown')}."
    )
    state["abstained"] = True
    state.setdefault("trace", []).append({"step": "abstain", "reason": state.get("grade_reason")})
    return state
