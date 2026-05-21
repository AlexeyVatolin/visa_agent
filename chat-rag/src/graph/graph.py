from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    classify_question,
    detect_country,
    generate_answer,
    input_guard,
    load_official_data,
    output_guard,
    reject,
    retrieve_from_chat,
)
from graph.state import GraphState


def route_after_input_guard(state: GraphState) -> str:
    if state.get("injection_flagged"):  # type: ignore[call-overload]
        return "reject"
    return "classify_question"


def route_after_classify(state: GraphState) -> str:
    if state["classification"] == "relevant":
        return "detect_country"
    return "reject"


def route_after_detect(state: GraphState) -> str:
    if state.get("country"):
        return ["retrieve_from_chat", "load_official_data"]
    return "generate_answer"


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("input_guard", input_guard)
    graph.add_node("classify_question", classify_question)
    graph.add_node("detect_country", detect_country)
    graph.add_node("retrieve_from_chat", retrieve_from_chat)
    graph.add_node("load_official_data", load_official_data)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("output_guard", output_guard)
    graph.add_node("reject", reject)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", route_after_input_guard)
    graph.add_conditional_edges("classify_question", route_after_classify)
    graph.add_conditional_edges("detect_country", route_after_detect)

    graph.add_edge("retrieve_from_chat", "generate_answer")
    graph.add_edge("load_official_data", "generate_answer")

    graph.add_edge("generate_answer", "output_guard")
    graph.add_edge("output_guard", END)
    graph.add_edge("reject", END)

    return graph.compile()
