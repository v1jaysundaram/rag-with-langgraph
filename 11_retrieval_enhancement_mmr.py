from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from typing_extensions import List, TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()


# ---------------------------
# INDEXING
# ---------------------------

embeddings = OpenAIEmbeddings()
persist_path = "rag_embeddings"

vector_store = FAISS.load_local(
    folder_path=persist_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# MMR retriever: lambda_mult=0 → maximally diverse results
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "lambda_mult": 1} # 1 - least  diverse, 0 - most diverse
)


# ---------------------------
# LLM
# ---------------------------

llm = ChatOpenAI(model_name="gpt-4o-mini")


# ---------------------------
# STATE
# ---------------------------

class RagState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    context: str
    augmented_query: str
    answer: str


# ---------------------------
# PROMPT
# ---------------------------

prompt_rag = PromptTemplate(
    template="""
You are an expert chat assistant designed to answer questions based on provided context.
Carefully use the context below to answer the question.
If the context does not contain enough information, respond clearly that you do not know the answer.

Question: {query}
Context: {context}
""",
    input_variables=["context", "query"]
)


# ---------------------------
# NODES
# ---------------------------

def retrieve(state: RagState):
    docs = retriever.invoke(state["query"])
    return {"retrieved_docs": docs}


def augment(state: RagState):
    context = "\n\n".join(doc.page_content for doc in state["retrieved_docs"])
    augmented_query = prompt_rag.format(query=state["query"], context=context)
    return {"context": context, "augmented_query": augmented_query}


def generate(state: RagState):
    response = llm.invoke(state["augmented_query"])
    return {"answer": response.content}


# ---------------------------
# GRAPH
# ---------------------------

graph = StateGraph(RagState)

graph.add_node("retrieve", retrieve)
graph.add_node("augment", augment)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "augment")
graph.add_edge("augment", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()


# ---------------------------
# TEST RUN
# ---------------------------

input_state = {"query": "What are the steps involved in training a language model?"}

response = workflow.invoke(input_state)
print(response["answer"])
