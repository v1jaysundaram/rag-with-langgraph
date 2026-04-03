from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor
from typing_extensions import List, TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()


# ---------------------------
# LOAD EXISTING VECTOR STORE
# ---------------------------

embeddings = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    folder_path="rag_embeddings",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)


# ---------------------------
# COMPRESSION RETRIEVER SETUP
# ---------------------------

base_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

compressor_llm = ChatOpenAI(model_name="gpt-4o-mini")
compressor = LLMChainExtractor.from_llm(compressor_llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)


# ---------------------------
# LLM
# ---------------------------

llm = ChatOpenAI(model_name="gpt-4o")


# ---------------------------
# STATE
# ---------------------------

class RagState(TypedDict):
    query: str
    compressed_docs: List[Document]
    context: str
    augmented_query: str
    answer: str


# ---------------------------
# PROMPTS
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
    query = state["query"]

    compressed_docs = compression_retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in compressed_docs)

    return {
        "compressed_docs": compressed_docs,
        "context": context
    }


def augment(state: RagState):
    augmented_query = prompt_rag.format(query=state["query"], context=state["context"])
    return {"augmented_query": augmented_query}


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

