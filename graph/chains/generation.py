from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langsmith import Client

client = Client()

prompt = client.pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True
)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

generation_chain = prompt | llm | StrOutputParser()
