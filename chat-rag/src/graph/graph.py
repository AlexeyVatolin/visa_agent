from langgraph.graph import END, START, StateGraph

from graph.state import GraphState
from graph.nodes import classify_question, reject, retrieve_from_chat, load_official_data, generate_answer


def route_after_classify(state: GraphState) -> list[str] | str:
    if state["classification"] == "relevant":
        return ["retrieve_from_chat", "load_official_data"]
    return "reject"


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("classify_question", classify_question)
    graph.add_node("retrieve_from_chat", retrieve_from_chat)
    graph.add_node("load_official_data", load_official_data)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("reject", reject)

    graph.add_edge(START, "classify_question")
    graph.add_conditional_edges("classify_question", route_after_classify)

    graph.add_edge("retrieve_from_chat", "generate_answer")
    graph.add_edge("load_official_data", "generate_answer")

    graph.add_edge("generate_answer", END)
    graph.add_edge("reject", END)

    return graph.compile()
