from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, True or False"
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer, method="json_mode")

system = """You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score True or False. True means that the answer resolves the question.
Return a JSON with a single key 'binary_score' and value True or False."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
