from typing import Any, Dict
from graph.state import GraphState

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    retriever = state["retriever"]  # gets retriever from state
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
