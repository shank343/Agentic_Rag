from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from tavily import TavilyClient

from graph.state import GraphState

# Load env vars
load_dotenv()

# Native Tavily client
tavily_client = TavilyClient()

def dedupe_documents(documents):
    seen = set()
    unique_docs = []

    for doc in documents:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)
    return unique_docs

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    
    question = state["question"]
    documents = state.get("documents", None)

    # 🔍 Tavily search
    response = tavily_client.search(
        query=question,
        max_results=3,
        search_depth="fast",   # 👈 as requested
        include_images=False   # 👈 still important
    )

    # 🧹 Clean results (no truncation)
    contents = []
    urls = []

    for r in response.get("results", []):
        contents.append(r.get("content", "")[:1000])
        urls.append(r.get("url", ""))

    joined_tavily_result = "\n".join(contents)

    # 📄 Convert to LangChain Document
    web_results = Document(
        page_content=joined_tavily_result,
        metadata={"sources": urls},
    )

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {"documents": dedupe_documents(documents), "question": question}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})