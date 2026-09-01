"""Retrieve-then-generate pipelines with no grading and no self-correction.
These are the comparison points the corrective loop is measured against.

Two variants, because the first version of this file quietly invalidated the
whole experiment. It reused GENERATE_SYSTEM_PROMPT from the agent, which ends
with "If the context does not fully support an answer, say so explicitly
rather than guessing" — so the "naive" baseline was already instructed to
stay grounded, and unsurprisingly it never hallucinated. That made the
headline comparison meaningless: both arms shared the intervention.

  grounded=True  -> same generation prompt as the agent. Isolates the
                    grade/reformulate/abstain LOOP as the only difference.
  grounded=False -> a genuinely naive prompt with no groundedness
                    instruction. This is the real "what happens if you just
                    stuff chunks in a prompt" baseline.

Keeping both arms is the point: it separates how much of the win comes from
one sentence of prompting versus the whole state machine."""
from google.genai import types

from src.agent.llm_client import generate
from src.agent.nodes import GENERATE_SYSTEM_PROMPT
from src.config import settings
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank


NAIVE_SYSTEM_PROMPT = """Answer the user's question using the provided context. Cite the source of every claim inline using the format [source_file]."""


def naive_answer(question: str, grounded: bool = True) -> dict:
    hits = hybrid_search(question, top_k=settings.top_k_retrieve)
    top_chunks = rerank(question, hits, top_k=settings.top_k_rerank)
    context = "\n\n".join(f"[{c.source}] {c.text}" for c in top_chunks)

    response = generate(
        model=settings.generator_model,
        contents=f"Question: {question}\n\nContext:\n{context}",
        config=types.GenerateContentConfig(
            system_instruction=GENERATE_SYSTEM_PROMPT if grounded else NAIVE_SYSTEM_PROMPT,
            max_output_tokens=800,
        ),
        fallback_model=settings.grader_model,
    )
    return {
        "answer": response.text.strip(),
        "sources": sorted({c.source for c in top_chunks}),
        # Raw chunk texts, kept so the RAGAS scorer can measure faithfulness
        # and context precision/recall against what was actually retrieved.
        "contexts": [c.text for c in top_chunks],
    }
