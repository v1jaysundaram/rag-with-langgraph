from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

load_dotenv()


# ---------------------------
# INDEXING WITH CCH
# ---------------------------

embeddings = OpenAIEmbeddings()
persist_path = "rag_embeddings_cch"

llm_cch = ChatOpenAI(model_name="gpt-4o-mini")

prompt_cch = PromptTemplate(
    template="""You are a document analysis assistant.
Given the following text chunk from a document, generate:
1. A short, descriptive title (max 10 words)
2. A 1-2 sentence summary of what this chunk is about

Respond in this exact format:
Title: <title>
Summary: <summary>

Chunk:
{chunk}""",
    input_variables=["chunk"]
)


def build_cch_index():
    file_path = os.path.join(os.path.dirname(__file__), "llm-book.pdf")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    enriched_chunks = []
    for chunk in chunks:
        response = llm_cch.invoke(prompt_cch.format(chunk=chunk.page_content))
        header = response.content.strip()
        enriched_content = f"{header}\n\n{chunk.page_content}"
        enriched_chunks.append(Document(page_content=enriched_content, metadata=chunk.metadata))

    vector_store = FAISS.from_documents(enriched_chunks, embedding=embeddings)
    vector_store.save_local(persist_path)
    return vector_store


if os.path.exists(persist_path):
    vector_store = FAISS.load_local(
        folder_path=persist_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vector_store = build_cch_index()


# ---------------------------
# LLM
# ---------------------------

llm = ChatOpenAI(model_name="gpt-4o")


# ---------------------------
# STATE
# ---------------------------

class RagState(TypedDict):
    query: str
    docs: List[Document]
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(state["query"])

    context = "\n\n".join(doc.page_content for doc in docs)

    return {"docs": docs, "context": context}


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
