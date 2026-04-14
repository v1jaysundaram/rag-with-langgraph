import re
from typing import List, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


# ---------------------------
# Vector store + retriever
# ---------------------------
embeddings = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    folder_path="rag_embeddings",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# ---------------------------
# LLMs
# ---------------------------
llm_mini = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)


# ---------------------------
# Thresholds
# ---------------------------
UPPER_TH = 0.7
LOWER_TH = 0.3


# ---------------------------
# State
# ---------------------------
class CragState(TypedDict):
    query: str
    docs: List[Document]        # raw retrieved docs
    good_docs: List[Document]   # docs scoring > LOWER_TH
    verdict: str                # CORRECT / AMBIGUOUS / INCORRECT
    web_query: str              # rewritten query for web search
    web_docs: List[Document]    # Tavily search results
    refined_context: str        # sentence-filtered final context
    answer: str


# ---------------------------
# Pydantic models
# ---------------------------
class DocScore(BaseModel):
    score: float
    reason: str


class WebQuery(BaseModel):
    query: str


class KeepOrDrop(BaseModel):
    keep: bool


# ---------------------------
# Prompts
# ---------------------------
doc_eval_prompt = (
    "You are a strict retrieval evaluator for RAG.\n"
    "Given ONE retrieved chunk and a question, return a relevance score in [0.0, 1.0].\n"
    "- 1.0: chunk alone is sufficient to answer fully\n"
    "- 0.0: chunk is completely irrelevant\n"
    "Be conservative with high scores. Also return a short reason.\n\n"
    "Question: {question}\n\nChunk:\n{chunk}"
)

rewrite_prompt = (
    "Rewrite the user question into a short web search query (6–14 words).\n"
    "Do NOT answer the question. Return JSON with a single key: query\n\n"
    "Question: {question}"
)

filter_prompt = (
    "Return keep=true only if the sentence directly helps answer the question.\n\n"
    "Question: {question}\n\nSentence:\n{sentence}"
)

answer_prompt = (
    "You are a helpful assistant. Answer ONLY using the provided context.\n"
    "If the context is empty or insufficient, say: 'I don't know.'\n\n"
    "Question: {question}\n\nContext:\n{context}"
)


# ---------------------------
# Tavily web search
# ---------------------------
tavily = TavilySearchResults(max_results=3)


# ---------------------------
# Helper: sentence splitter
# ---------------------------
def split_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ---------------------------
# Nodes
# ---------------------------
def retrieve(state: CragState):
    docs = retriever.invoke(state["query"])
    return {"docs": docs}


def eval_docs(state: CragState):
    q = state["query"]
    scores = []
    good = []

    for doc in state["docs"]:
        result = llm_mini.with_structured_output(DocScore).invoke(
            doc_eval_prompt.format(question=q, chunk=doc.page_content)
        )
        scores.append(result.score)
        if result.score > LOWER_TH:
            good.append(doc)

    if any(s > UPPER_TH for s in scores):
        return {"good_docs": good, "verdict": "CORRECT"}

    if scores and all(s < LOWER_TH for s in scores):
        return {"good_docs": [], "verdict": "INCORRECT"}

    return {"good_docs": good, "verdict": "AMBIGUOUS"}


def rewrite_query(state: CragState):
    result = llm_mini.with_structured_output(WebQuery).invoke(
        rewrite_prompt.format(question=state["query"])
    )
    return {"web_query": result.query}


def web_search(state: CragState):
    q = state.get("web_query") or state["query"]
    results = tavily.invoke({"query": q})

    web_docs = []
    for r in results or []:
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {r.get('title', '')}\nURL: {r.get('url', '')}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": r.get("url", "")}))

    return {"web_docs": web_docs}


def refine(state: CragState):
    q = state["query"]
    verdict = state["verdict"]

    if verdict == "CORRECT":
        docs_to_use = state["good_docs"]
    elif verdict == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:  # AMBIGUOUS
        docs_to_use = state["good_docs"] + state["web_docs"]

    raw_context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    sentences = split_to_sentences(raw_context)

    kept = []
    for s in sentences:
        if llm_mini.with_structured_output(KeepOrDrop).invoke(
            filter_prompt.format(question=q, sentence=s)
        ).keep:
            kept.append(s)
    return {"refined_context": "\n".join(kept).strip()}


def generate(state: CragState):
    response = llm.invoke(
        answer_prompt.format(question=state["query"], context=state["refined_context"])
    )
    return {"answer": response.content}


# ---------------------------
# Router
# ---------------------------
def route_after_eval(state: CragState) -> Literal["refine", "rewrite_query"]:
    if state["verdict"] == "CORRECT":
        return "refine"
    return "rewrite_query"


# ---------------------------
# Graph
# ---------------------------
graph = StateGraph(CragState)

graph.add_node("retrieve", retrieve)
graph.add_node("eval_docs", eval_docs)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("refine", refine)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "eval_docs")

graph.add_conditional_edges(
    "eval_docs",
    route_after_eval,
    {"refine": "refine", "rewrite_query": "rewrite_query"},
)

graph.add_edge("rewrite_query", "web_search")
graph.add_edge("web_search", "refine")
graph.add_edge("refine", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()


# ---------------------------
# Test run
# ---------------------------
input_state = {
    "query": "What is claude code?",
    "docs": [],
    "good_docs": [],
    "verdict": "",
    "web_query": "",
    "web_docs": [],
    "refined_context": "",
    "answer": "",
}

response = workflow.invoke(input_state)

print(response["answer"])
