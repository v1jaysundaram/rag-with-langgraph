from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


# ---------------------------
# INDEXING
# ---------------------------

embeddings = OpenAIEmbeddings()
persist_path = "rag_embeddings_reranking"

def build_reranking_index():
    file_path = os.path.join(os.path.dirname(__file__), "hr-policy.pdf")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    vector_store.save_local(persist_path)
    return vector_store

if os.path.exists(persist_path):
    vector_store = FAISS.load_local(
        folder_path=persist_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vector_store = build_reranking_index()


# ---------------------------
# LLMs
# ---------------------------

reranker_llm = ChatOpenAI(model_name="gpt-4o-mini")
llm = ChatOpenAI(model_name="gpt-4o")


# ---------------------------
# STATE
# ---------------------------

class RagState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    context: str
    augmented_query: str
    answer: str


# ---------------------------
# OUTPUT PARSER
# ---------------------------

class RelevanceScore(BaseModel):
    score: int = Field(description="Relevance score between 1 and 10")

rerank_parser = PydanticOutputParser(pydantic_object=RelevanceScore)


# ---------------------------
# PROMPTS
# ---------------------------

prompt_rerank = PromptTemplate(
    template="""On a scale of 1-10, rate the relevance of the following document to the query.
Consider the specific context and intent of the query, not just keyword matches.

Query: {query}
Document: {document}

{format_instructions}""",
    input_variables=["query", "document"],
    partial_variables={"format_instructions": rerank_parser.get_format_instructions()}
)

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
    results = vector_store.similarity_search(state["query"], k=6)
    return {"retrieved_docs": results}


def rerank(state: RagState):
    query = state["query"]
    docs = state["retrieved_docs"]

    scored_docs = []
    for doc in docs:
        prompt = prompt_rerank.format(query=query, document=doc.page_content)
        response = reranker_llm.invoke(prompt)
        parsed = rerank_parser.parse(response.content)
        scored_docs.append((parsed.score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    reranked_docs = [doc for _, doc in scored_docs[:4]]

    return {"reranked_docs": reranked_docs}


def augment(state: RagState):
    context = "\n\n".join(doc.page_content for doc in state["reranked_docs"])
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
graph.add_node("rerank", rerank)
graph.add_node("augment", augment)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "augment")
graph.add_edge("augment", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()


# ---------------------------
# TEST RUN
# ---------------------------

input_state = {"query": "Is the appointment letter given after the probation period?"}

response = workflow.invoke(input_state)
print(response["answer"])
