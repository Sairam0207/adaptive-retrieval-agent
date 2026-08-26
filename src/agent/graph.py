"""LangGraph state machine wiring the retrieve -> grade -> correct loop."""
from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    abstain_node,
    generate_node,
    grade_node,
    reformulate_node,
    retrieve_node,
    route_after_grade,
)
from src.agent.state import AgentState
from src.config import settings


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("generate", generate_node)
    graph.add_node("abstain", abstain_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "reformulate": "reformulate", "abstain": "abstain"},
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("abstain", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(question: str) -> AgentState:
    graph = get_compiled_graph()
    initial_state: AgentState = {
        "query": question,
        "original_query": question,
        "retry_count": 0,
        "max_retries": settings.max_retries,
        "trace": [],
    }
    return graph.invoke(initial_state)
