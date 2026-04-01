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

# INDEXING

embeddings = OpenAIEmbeddings()
persist_path = "rag_embeddings_cwe"

def build_cwe_index():
    file_path = os.path.join(os.path.dirname(__file__), "llm-book.pdf")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = text_splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

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
    vector_store = build_cwe_index()

# LLM

llm = ChatOpenAI(model_name="gpt-4o")

# STATE

class RagState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    docs: List[Document]
    context: str
    augmented_query: str
    answer: str

# PROMPTS

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

# NODES

def retrieve(state: RagState):
    results = vector_store.similarity_search(state["query"], k=4)

    expanded_indices = set()
    for doc in results:
        idx = doc.metadata["chunk_index"]
        expanded_indices.update([idx - 1, idx, idx + 1])

    all_docs = vector_store.docstore._dict.values()
    expanded_docs = sorted(
        [doc for doc in all_docs if doc.metadata["chunk_index"] in expanded_indices],
        key=lambda doc: doc.metadata["chunk_index"]
    )

    context = "\n\n".join(doc.page_content for doc in expanded_docs)
    return {"retrieved_docs": results, "docs": expanded_docs, "context": context}

def augment(state: RagState):
    augmented_query = prompt_rag.format(query=state["query"], context=state["context"])
    return {"augmented_query": augmented_query}

def generate(state: RagState):
    response = llm.invoke(state["augmented_query"])
    return {"answer": response.content}

# GRAPH

graph = StateGraph(RagState)

graph.add_node("retrieve", retrieve)
graph.add_node("augment", augment)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "augment")
graph.add_edge("augment", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()

# TEST RUN

input_state = {"query": "What are the applications of an LLM?"}
response = workflow.invoke(input_state)
print(response["answer"])
