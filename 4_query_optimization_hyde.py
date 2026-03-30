from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# load docs from FAISS
embeddings = OpenAIEmbeddings()

persist_path = "rag_embeddings"

vector_store = FAISS.load_local(
    folder_path=persist_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# LLM
llm = ChatOpenAI(model_name="gpt-4o-mini")

# ---------------------------
# STATE
# ---------------------------

class RagState(TypedDict):
    query: str
    hypothetical_doc: str
    chunk_size: int
    docs: List[Document]
    context: str
    augmented_query: str
    answer: str


# ---------------------------
# PROMPTS
# ---------------------------

# RAG prompt 
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


# HyDE prompt
prompt_hyde = PromptTemplate(
    template="""
You are part of a Retrieval-Augmented Generation (RAG) system.

Your task is to generate a hypothetical document that would likely exist in the knowledge base and directly answer the user's question.

This document will be embedded and used for semantic retrieval.

Requirements:
- The document must directly answer the question.
- It should not mention that it is hypothetical.
- It must be exactly {chunk_size} characters long.
- It should resemble real knowledge base content to improve embedding similarity.

User Question:
{query}
""",
    input_variables=["query", "chunk_size"]
)

# ---------------------------
# NODES
# ---------------------------

def hyde(state: RagState):
    response = llm.invoke(prompt_hyde.format(query=state["query"], chunk_size=state["chunk_size"]))

    hypothetical_doc = response.content[:state["chunk_size"]]

    return {"hypothetical_doc": hypothetical_doc}


def retrieve(state: RagState):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(state["hypothetical_doc"])

    context = "\n\n".join(doc.page_content for doc in docs)

    return {
        "docs": docs,
        "context": context
    }


def augment(state: RagState):
    augmented_query = prompt_rag.format(
        query=state["query"],
        context=state["context"]
    )

    return {"augmented_query": augmented_query}


def generate(state: RagState):
    response = llm.invoke(state["augmented_query"])
    return {"answer": response.content}


# ---------------------------
# GRAPH
# ---------------------------

graph = StateGraph(RagState)

graph.add_node("hyde", hyde)
graph.add_node("retrieve", retrieve)
graph.add_node("augment", augment)
graph.add_node("generate", generate)

graph.add_edge(START, "hyde")
graph.add_edge("hyde", "retrieve")
graph.add_edge("retrieve", "augment")
graph.add_edge("augment", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()


# ---------------------------
# TEST RUN
# ---------------------------

input_state = {
    "query": "What does large in large language model mean?",
    "chunk_size": 400
}

response = workflow.invoke(input_state)

print(response["answer"])
