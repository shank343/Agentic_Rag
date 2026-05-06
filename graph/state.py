from typing import List, TypedDict, Any


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
    retriever: Any
    route_score: float
    route_decision: str
    generation_count: int
