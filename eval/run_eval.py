"""Compares the naive baseline against the corrective agent on a hand-built
golden set, using an LLM-as-judge for fact coverage / faithfulness / hallucination.
This produces the numbers that back up the project's core claim: self-correction
measurably reduces unsupported answers.

Usage: python -m eval.run_eval
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from google.genai import types

from src.agent.graph import ask
from src.agent.llm_client import generate
from src.agent.naive_baseline import naive_answer
from src.config import settings

JUDGE_SYSTEM_PROMPT = """You are grading a RAG system's answer against a question and a \
list of facts the answer is expected to cover. Respond with ONLY a JSON object:
{
  "fact_coverage": <float 0.0-1.0, fraction of expected facts the answer actually covers>,
  "faithfulness": <int 1-5, is the answer grounded / not fabricated, 5 = fully grounded>,
  "hallucinated": <true/false, true if the answer confidently states something false or \
unsupported instead of expressing uncertainty when it should have>
}"""


def judge(question: str, expected_facts: list[str], answer: str) -> dict:
    response = generate(
        model=settings.grader_model,
        contents=(
            f"Question: {question}\n"
            f"Expected facts: {expected_facts}\n"
            f"Answer to grade: {answer}"
        ),
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT,
            max_output_tokens=200,
            response_mime_type="application/json",
        ),
    )
    raw = response.text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"fact_coverage": 0.0, "faithfulness": 1, "hallucinated": True}


CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "results_checkpoint.json")


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"naive": [], "agent": []}


def _save_checkpoint(results: dict) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run() -> None:
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    results = _load_checkpoint()
    done_ids = {r["id"] for r in results["agent"]}

    for item in dataset:
        if item["id"] in done_ids:
            continue
        print(f"Evaluating {item['id']} ({item['type']})...")

        naive_result = naive_answer(item["question"])
        naive_score = judge(item["question"], item["expected_facts"], naive_result["answer"])
        results["naive"].append({**naive_score, "type": item["type"], "id": item["id"]})

        agent_result = ask(item["question"])
        agent_score = judge(item["question"], item["expected_facts"], agent_result["answer"])
        agent_score["retries_used"] = agent_result["retry_count"]
        agent_score["abstained"] = agent_result["abstained"]
        results["agent"].append({**agent_score, "type": item["type"], "id": item["id"]})

        _save_checkpoint(results)

    _report(results)


def _avg(values: list[dict], key: str) -> float:
    return round(sum(v[key] for v in values) / len(values), 3) if values else 0.0


def _report(results: dict) -> None:
    print("\n" + "=" * 60)
    print("EVAL RESULTS: naive baseline vs corrective agent")
    print("=" * 60)

    # fact_coverage only means something for questions the corpus can actually
    # answer. On unanswerable questions, a high fact_coverage score just means
    # the model asserted a true claim from its own general knowledge rather
    # than from the retrieved context — which is exactly the kind of ungrounded
    # answer a RAG system is supposed to avoid, not something to reward. So
    # that subset is reported separately: what matters there is whether the
    # system fabricated an unsupported answer or correctly declined.
    for system_name, rows in results.items():
        answerable = [r for r in rows if r["type"] != "unanswerable"]
        unanswerable = [r for r in rows if r["type"] == "unanswerable"]

        print(f"\n[{system_name}] answerable questions (n={len(answerable)})")
        print(f"  avg fact_coverage: {_avg(answerable, 'fact_coverage')}")
        print(f"  avg faithfulness:  {_avg(answerable, 'faithfulness')}")
        print(f"  hallucination rate: {sum(1 for r in answerable if r['hallucinated']) / len(answerable):.0%}")

        if unanswerable:
            halluc_on_unanswerable = sum(1 for r in unanswerable if r["hallucinated"])
            print(f"[{system_name}] unanswerable questions (n={len(unanswerable)})")
            print(f"  hallucinated (fabricated an unsupported claim): {halluc_on_unanswerable}/{len(unanswerable)}")

    agent_retries = [r["retries_used"] for r in results["agent"]]
    agent_abstains = sum(1 for r in results["agent"] if r["abstained"])
    agent_answerable = [r for r in results["agent"] if r["type"] != "unanswerable"]
    agent_abstains_on_answerable = sum(1 for r in agent_answerable if r["abstained"])
    print(f"\n[agent] avg retries used: {sum(agent_retries) / len(agent_retries):.2f}")
    print(f"[agent] abstained on {agent_abstains}/{len(results['agent'])} questions overall")
    print(
        f"[agent] abstained on {agent_abstains_on_answerable}/{len(agent_answerable)} "
        "questions the corpus could actually answer (over-conservative grading cost)"
    )


if __name__ == "__main__":
    run()
