"""RAGAS scoring for the three arms, alongside the hand-rolled LLM judge.

Why this exists
---------------
The hand-rolled judge in run_eval.py asks one model a yes/no question:
"did this answer hallucinate?" That flag turned out to be too coarse to
separate the arms — it returned 0 hallucinations for *every* system on
*every* question, including ones the corpus cannot answer. A metric that
never fires cannot tell you which system is better.

RAGAS measures groundedness differently, and that difference is the point:

  faithfulness          decomposes the answer into individual claims and
                        checks each one against the retrieved context. Graded
                        0-1, not binary, so partial ungroundedness shows up.
  answer_relevancy      does the answer actually address the question, or is
                        it evasive? This is what catches an agent that games
                        a groundedness metric by abstaining on everything.
  context_precision     of the chunks retrieved, how many were relevant?
                        Measures the retriever, not the generator.
  context_recall        did retrieval find everything the reference needs?

faithfulness and answer_relevancy pull in opposite directions, which is
exactly why both are here: abstaining scores perfectly on faithfulness (no
claims, nothing to contradict) and terribly on answer_relevancy. Reporting
either one alone would flatter this system.

Abstentions are scored separately rather than averaged in. An abstention has
no claims to verify, so folding it into a faithfulness mean silently rewards
a system for declining to answer. See _split_abstentions.

Usage:
    python -m eval.ragas_eval                    # all arms, all questions
    python -m eval.ragas_eval --limit 4          # smoke test
    python -m eval.ragas_eval --arms agent       # one arm only
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.graph import ask
from src.agent.naive_baseline import naive_answer
from src.config import settings

SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "ragas_samples.json")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")

# Phrase the abstain node emits. Kept in sync with abstain_node in
# src/agent/nodes.py — if that wording changes, this must too.
ABSTAIN_MARKER = "don't have enough information"

ARMS = {
    # arm name        -> how to produce (answer, contexts) for a question
    "naive_ungrounded": lambda q: _from_naive(q, grounded=False),
    "naive_grounded": lambda q: _from_naive(q, grounded=True),
    "agent": lambda q: _from_agent(q),
}


def _from_naive(question: str, grounded: bool) -> dict:
    result = naive_answer(question, grounded=grounded)
    return {"answer": result["answer"], "contexts": result["contexts"]}


def _from_agent(question: str) -> dict:
    state = ask(question)
    return {
        "answer": state["answer"],
        "contexts": [c.text for c in state.get("retrieved_chunks", [])],
        "abstained": state.get("abstained", False),
    }


def _load(path: str, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def build_samples(arms: list[str], limit: int | None) -> dict:
    """Generate (answer, contexts) per arm per question, caching as it goes.

    Generation is the expensive half and the free-tier quota is the binding
    constraint, so this checkpoints after every single sample: a run killed by
    a quota wall can be resumed tomorrow without paying for what it already
    has.
    """
    dataset = _load(DATASET_PATH, [])
    if limit:
        dataset = dataset[:limit]
    samples = _load(SAMPLES_PATH, {})

    for arm in arms:
        samples.setdefault(arm, {})
        for item in dataset:
            if item["id"] in samples[arm]:
                continue
            print(f"  generating {arm}/{item['id']} ({item['type']})...", flush=True)
            produced = ARMS[arm](item["question"])
            samples[arm][item["id"]] = {
                "question": item["question"],
                "type": item["type"],
                "subtype": item.get("subtype"),
                "reference": " ".join(item["expected_facts"]),
                **produced,
            }
            with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
    return samples


def _is_abstention(row: dict) -> bool:
    return bool(row.get("abstained")) or ABSTAIN_MARKER in row["answer"].lower()


def _split_abstentions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """An abstention has no factual claims, so faithfulness is vacuously
    perfect on it. Averaging those in would let a system that abstains on
    everything post a 1.0 faithfulness score. Answered and abstained rows are
    therefore reported as separate populations, never blended."""
    answered = [r for r in rows if not _is_abstention(r)]
    abstained = [r for r in rows if _is_abstention(r)]
    return answered, abstained


def _build_evaluator():
    """Gemini as the judge LLM; the project's own local bge-small model for
    embeddings. Using the local embedder keeps answer_relevancy off the API
    quota entirely and scores relevancy in the same vector space the retriever
    actually searches in."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=settings.grader_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model)
    )
    return llm, embeddings


def score(samples: dict, arms: list[str]) -> dict:
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    llm, embeddings = _build_evaluator()
    metrics = [
        Faithfulness(llm=llm),
        # strictness=1 is required, not a tuning choice. ResponseRelevancy
        # defaults to strictness=3, which asks the LLM for 3 candidates in one
        # call (n=3). Gemini rejects that outright — "Multiple candidates is
        # not enabled for this model" (400 INVALID_ARGUMENT) — and RAGAS
        # swallows the error and reports nan rather than failing loudly.
        ResponseRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]
    # Free-tier quotas are per-minute and shared across call sites, so RAGAS
    # must run strictly serially. Its default fan-out would trip a 429 within
    # seconds and poison a whole batch of scores.
    run_config = RunConfig(max_workers=1, timeout=180)

    report = {}
    for arm in arms:
        rows = list(samples.get(arm, {}).values())
        if not rows:
            continue
        answered, abstained = _split_abstentions(rows)
        print(f"\nScoring [{arm}]: {len(answered)} answered, {len(abstained)} abstained")
        if not answered:
            report[arm] = {"answered": 0, "abstained": len(abstained)}
            continue

        dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["reference"],
            )
            for r in answered
        ])
        result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
        report[arm] = {
            "answered": len(answered),
            "abstained": len(abstained),
            "abstention_rate": round(len(abstained) / len(rows), 3),
            "scores": {k: round(v, 3) for k, v in result._repr_dict.items()
                       if isinstance(v, (int, float))},
        }
    return report


def _print_report(report: dict) -> None:
    print("\n" + "=" * 66)
    print("RAGAS RESULTS (scored on answered questions only)")
    print("=" * 66)
    for arm, data in report.items():
        print(f"\n[{arm}]")
        print(f"  answered {data['answered']}, abstained {data['abstained']} "
              f"(abstention rate {data.get('abstention_rate', 0)})")
        for metric, value in data.get("scores", {}).items():
            print(f"  {metric:<28} {value}")
    print(
        "\nRead faithfulness and answer_relevancy together: abstaining inflates "
        "the first and destroys the second, so neither number means much alone."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N questions (smoke test)")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--score-only", action="store_true",
                        help="skip generation, score whatever is already cached")
    args = parser.parse_args()

    if args.score_only:
        samples = _load(SAMPLES_PATH, {})
    else:
        print("Generating answers (resumable; cached to eval/ragas_samples.json)")
        samples = build_samples(args.arms, args.limit)

    report = score(samples, args.arms)
    _print_report(report)

    out = os.path.join(os.path.dirname(__file__), "ragas_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
