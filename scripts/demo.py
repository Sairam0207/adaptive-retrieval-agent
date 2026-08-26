"""Interactive CLI to watch the retrieve/grade/correct/answer loop live."""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agent.graph import ask


def print_trace(trace: list[dict]) -> None:
    for entry in trace:
        step = entry["step"]
        if step == "retrieve":
            print(f"  -> retrieve: query='{entry['query']}' hits={entry['num_hits']}")
        elif step == "grade":
            print(f"  -> grade: verdict={entry['verdict']} reason='{entry['reason']}'")
        elif step == "reformulate":
            print(f"  -> reformulate (retry {entry['retry_count']}): "
                  f"'{entry['old_query']}' -> '{entry['new_query']}'")
        elif step == "generate":
            print("  -> generate: answer produced")
        elif step == "abstain":
            print(f"  -> abstain: {entry['reason']}")


def main() -> None:
    print("Adaptive Retrieval Agent — type a question (or 'quit').\n")
    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        result = ask(question)
        print("\n--- trace ---")
        print_trace(result["trace"])
        print("\n--- answer ---")
        print(result["answer"])
        print(f"\n(retries used: {result['retry_count']}, abstained: {result['abstained']})\n")


if __name__ == "__main__":
    main()
