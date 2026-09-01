"""Regression guard for the bug that invalidated the first eval.

naive_baseline.py originally reused the agent's GENERATE_SYSTEM_PROMPT, which
instructs the model to decline when context is insufficient. That handed the
control arm the exact intervention it was supposed to isolate, so the measured
difference between "naive" and "agent" meant nothing.

These tests fail if the two prompts ever converge again.
"""
from src.agent.naive_baseline import NAIVE_SYSTEM_PROMPT
from src.agent.nodes import GENERATE_SYSTEM_PROMPT

GROUNDEDNESS_CLAUSE = "say so explicitly rather than guessing"


def test_agent_prompt_instructs_groundedness() -> None:
    assert GROUNDEDNESS_CLAUSE in GENERATE_SYSTEM_PROMPT


def test_naive_prompt_does_not_instruct_groundedness() -> None:
    # The whole point of the ungrounded arm: no abstention hint of any kind.
    assert GROUNDEDNESS_CLAUSE not in NAIVE_SYSTEM_PROMPT
    assert "does not fully support" not in NAIVE_SYSTEM_PROMPT


def test_prompts_are_actually_different() -> None:
    assert NAIVE_SYSTEM_PROMPT != GENERATE_SYSTEM_PROMPT


def test_naive_arm_selects_the_right_prompt() -> None:
    """grounded= must actually route to a different system_instruction."""
    from unittest.mock import MagicMock, patch

    seen = []

    def fake_generate(model, contents, config, fallback_model=None):
        seen.append(config.system_instruction)
        response = MagicMock()
        response.text = "answer"
        return response

    with (
        patch("src.agent.naive_baseline.generate", side_effect=fake_generate),
        patch("src.agent.naive_baseline.hybrid_search", return_value=[]),
        patch("src.agent.naive_baseline.rerank", return_value=[]),
    ):
        from src.agent.naive_baseline import naive_answer

        naive_answer("q", grounded=True)
        naive_answer("q", grounded=False)

    assert seen == [GENERATE_SYSTEM_PROMPT, NAIVE_SYSTEM_PROMPT]
