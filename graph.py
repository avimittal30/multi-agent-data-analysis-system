"""
graph.py — LangGraph graph for query-first multi-agent system.

Flow:
  START → orchestrator → codegen → executor ─┬─(error, retries left)→ codegen (retry)
                                              └─(success / max retries)→ interpreter → END

  If engine == "direct": orchestrator → interpreter (skips codegen + executor)
"""
from langgraph.graph import StateGraph, START, END

from state import AgentState
from agents.nodes import (
    orchestrator_node,
    codegen_node,
    executor_node,
    interpreter_node,
    should_retry,
    increment_retry,
)


def build_graph():
    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────
    builder.add_node("orchestrator",  orchestrator_node)
    builder.add_node("codegen",       codegen_node)
    builder.add_node("executor",      executor_node)
    builder.add_node("increment_retry", increment_retry)
    builder.add_node("interpreter",   interpreter_node)

    # ── Edges ─────────────────────────────────────────────────────────────
    builder.add_edge(START, "orchestrator")

    # Orchestrator → codegen OR skip straight to interpreter (direct answers)
    builder.add_conditional_edges(
        "orchestrator",
        lambda s: "interpret" if s.get("engine") == "direct" else "codegen",
        {"codegen": "codegen", "interpret": "interpreter"},
    )

    builder.add_edge("codegen", "executor")

    # Executor → retry loop OR interpreter
    builder.add_conditional_edges(
        "executor",
        should_retry,
        {"retry": "increment_retry", "interpret": "interpreter"},
    )

    builder.add_edge("increment_retry", "codegen")   # loop back
    builder.add_edge("interpreter", END)

    return builder.compile()


graph = build_graph()
