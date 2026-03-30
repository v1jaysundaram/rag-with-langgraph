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
# INDEXING WITH HyPE
# ---------------------------

embeddings = OpenAIEmbeddings()
persist_path = "rag_embeddings_hype"

llm_hype = ChatOpenAI(model_name="gpt-4o-mini")

prompt_hype = PromptTemplate(
    template="""You are a document analysis assistant.
Given the following text chunk from a document, generate exactly 3 hypothetical questions
that this chunk could answer. These questions should cover the key ideas in the chunk.

Respond with ONLY the 3 questions, one per line, no numbering or bullet points.

Chunk:
{chunk}""",
    input_variables=["chunk"]
)


def build_hype_index():
    file_path = os.path.join(os.path.dirname(__file__), "llm-book.pdf")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    question_docs = []
    for chunk in chunks:
        response = llm_hype.invoke(prompt_hype.format(chunk=chunk.page_content))
        questions = [q.strip() for q in response.content.strip().splitlines() if q.strip()]
        for question in questions:
            question_docs.append(Document(
                page_content=question,
                metadata={**chunk.metadata, "original_content": chunk.page_content}
            ))

    vector_store = FAISS.from_documents(question_docs, embedding=embeddings)
    vector_store.save_local(persist_path)
    return vector_store


if os.path.exists(persist_path):
    vector_store = FAISS.load_local(
        folder_path=persist_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vector_store = build_hype_index()


# ---------------------------
# LLM
# ---------------------------

llm = ChatOpenAI(model_name="gpt-4o")


# ---------------------------
# STATE
# ---------------------------

class RagState(TypedDict):
    query: str
    matched_questions: List[str]
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    # Query matches against embedded hypothetical questions
    matched_docs = retriever.invoke(state["query"])

    # Extract original chunks from metadata, deduplicate by content
    matched_questions = [doc.page_content for doc in matched_docs]

    seen = set()
    docs = []
    for doc in matched_docs:
        original = doc.metadata.get("original_content", doc.page_content)
        if original not in seen:
            seen.add(original)
            docs.append(Document(page_content=original, metadata=doc.metadata))

    context = "\n\n".join(doc.page_content for doc in docs)

    return {"matched_questions": matched_questions, "docs": docs, "context": context}


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

input_state = {"query": "Differentiate between encoder and decoder."}

response = workflow.invoke(input_state)
print(response["answer"])
