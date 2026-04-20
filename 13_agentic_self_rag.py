from typing import List, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()


# ---------------------------
# Vector store + retriever
# ---------------------------
embeddings = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    folder_path="rag_embeddings_reranking",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------
# Limits
# ---------------------------
MAX_RETRIES = 3        # IsSUP revise loop
MAX_REWRITE_TRIES = 2  # IsUSE rewrite loop


# ---------------------------
# State
# ---------------------------
class RagState(TypedDict):
    question: str
    retrieval_query: str
    rewrite_tries: int

    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str

    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int

    isuse: Literal["useful", "not_useful"]
    use_reason: str


# ---------------------------
# Pydantic output schemas
# ---------------------------
class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(
        description="True if the question requires specific facts from documents, else False."
    )


class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        description="True if the document discusses the same topic as the question."
    )


class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)


class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str


class RewriteDecision(BaseModel):
    retrieval_query: str = Field(
        description="Rewritten query optimised for vector retrieval."
    )


# ---------------------------
# Prompts (plain strings)
# ---------------------------
decide_retrieval_prompt = (
    "Decide whether retrieval is needed to answer the question.\n"
    "Return JSON with key: should_retrieve (boolean).\n\n"
    "Rules:\n"
    "- True if the question asks for specific technical facts or details from documents.\n"
    "- False for simple factual or general knowledge questions.\n"
    "- When unsure, choose True.\n\n"
    "Question: {question}"
)

direct_generation_prompt = (
    "Answer the question using only your general knowledge.\n"
    "If it requires specific document details, say: 'I don't know based on general knowledge.'\n\n"
    "Question: {question}"
)

is_relevant_prompt = (
    "Judge whether the document is relevant to the question at a TOPIC level.\n"
    "Return JSON with key: is_relevant (boolean).\n\n"
    "A document is relevant if it discusses the same concept or topic area as the question.\n"
    "It does NOT need to contain the exact answer — that is checked later.\n"
    "When unsure, return is_relevant=true.\n\n"
    "Question: {question}\n\nDocument:\n{document}"
)

rag_generation_prompt = (
    "Answer the question using ONLY the provided context.\n"
    "Do not mention that you are using a context or any external source.\n\n"
    "Question: {question}\n\nContext:\n{context}"
)

issup_prompt = (
    "Verify whether the answer is supported by the context.\n"
    "Return JSON with keys: issup, evidence.\n"
    "issup must be one of: fully_supported, partially_supported, no_support.\n\n"
    "fully_supported: every claim is explicitly in the context, no added interpretation.\n"
    "partially_supported: core facts are present but the answer adds qualitative phrasing not in context.\n"
    "no_support: key claims are not in the context.\n\n"
    "Be strict — any unsupported qualitative phrasing → partially_supported.\n"
    "evidence: up to 3 short direct quotes from the context that support the answer.\n\n"
    "Question: {question}\n\nAnswer: {answer}\n\nContext:\n{context}"
)

revise_prompt = (
    "Rewrite the answer using ONLY direct quotes from the context.\n\n"
    "FORMAT:\n"
    "- <direct quote from context>\n"
    "- <direct quote from context>\n\n"
    "Rules:\n"
    "- Use ONLY the context. No added words, explanations, or interpretation.\n"
    "- Do NOT say 'context', 'not mentioned', 'not provided', etc.\n\n"
    "Question: {question}\n\nCurrent Answer: {answer}\n\nContext:\n{context}"
)

isuse_prompt = (
    "Judge whether the answer actually addresses the question.\n"
    "Return JSON with keys: isuse, reason.\n"
    "isuse must be one of: useful, not_useful.\n\n"
    "useful: the answer directly answers what was asked.\n"
    "not_useful: the answer is generic, off-topic, or only provides background.\n"
    "reason: one short line.\n\n"
    "Question: {question}\n\nAnswer: {answer}"
)

rewrite_for_retrieval_prompt = (
    "Rewrite the question into a short query optimised for vector retrieval (6–16 words).\n"
    "Keep key technical terms. Add 2–3 high-signal keywords likely in the source text.\n"
    "Remove filler words. Do NOT answer the question.\n"
    "Return JSON with key: retrieval_query.\n\n"
    "Question: {question}\n"
    "Previous retrieval query: {retrieval_query}\n"
    "Previous answer (if any): {answer}"
)


# ---------------------------
# Nodes
# ---------------------------
def decide_retrieval(state: RagState):
    decision: RetrieveDecision = llm.with_structured_output(RetrieveDecision).invoke(
        decide_retrieval_prompt.format(question=state["question"])
    )
    return {"need_retrieval": decision.should_retrieve}


def generate_direct(state: RagState):
    out = llm.invoke(direct_generation_prompt.format(question=state["question"]))
    return {"answer": out.content}


def retrieve(state: RagState):
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}


def is_relevant(state: RagState):
    relevant_docs: List[Document] = []
    for doc in state.get("docs", []):
        decision: RelevanceDecision = llm.with_structured_output(RelevanceDecision).invoke(
            is_relevant_prompt.format(
                question=state["question"],
                document=doc.page_content,
            )
        )
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}


def generate_from_context(state: RagState):
    context = "\n\n---\n\n".join(d.page_content for d in state.get("relevant_docs", [])).strip()
    if not context:
        return {"answer": "No answer found.", "context": ""}
    out = llm.invoke(rag_generation_prompt.format(question=state["question"], context=context))
    return {"answer": out.content, "context": context}


def no_answer_found(state: RagState):
    return {"answer": "No answer found.", "context": ""}


def is_sup(state: RagState):
    decision: IsSUPDecision = llm.with_structured_output(IsSUPDecision).invoke(
        issup_prompt.format(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"issup": decision.issup, "evidence": decision.evidence}


def revise_answer(state: RagState):
    out = llm.invoke(
        revise_prompt.format(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", ""),
        )
    )
    return {"answer": out.content, "retries": state.get("retries", 0) + 1}


def accept_answer(state: RagState):
    return {}


def is_use(state: RagState):
    decision: IsUSEDecision = llm.with_structured_output(IsUSEDecision).invoke(
        isuse_prompt.format(
            question=state["question"],
            answer=state.get("answer", ""),
        )
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}


def rewrite_question(state: RagState):
    decision: RewriteDecision = llm.with_structured_output(RewriteDecision).invoke(
        rewrite_for_retrieval_prompt.format(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", ""),
            answer=state.get("answer", ""),
        )
    )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }


# ---------------------------
# Routing
# ---------------------------
def route_after_decide(state: RagState) -> Literal["retrieve", "generate_direct"]:
    return "retrieve" if state["need_retrieval"] else "generate_direct"


def route_after_relevance(state: RagState) -> Literal["generate_from_context", "no_answer_found"]:
    if state.get("relevant_docs"):
        return "generate_from_context"
    return "no_answer_found"


def route_after_issup(state: RagState) -> Literal["accept_answer", "revise_answer"]:
    if state.get("issup") == "fully_supported" or state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"
    return "revise_answer"


def route_after_isuse(state: RagState) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"
    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"
    return "rewrite_question"


# ---------------------------
# Graph
# ---------------------------
graph = StateGraph(RagState)

graph.add_node("decide_retrieval", decide_retrieval)
graph.add_node("generate_direct", generate_direct)
graph.add_node("retrieve", retrieve)
graph.add_node("is_relevant", is_relevant)
graph.add_node("generate_from_context", generate_from_context)
graph.add_node("no_answer_found", no_answer_found)
graph.add_node("is_sup", is_sup)
graph.add_node("revise_answer", revise_answer)
graph.add_node("accept_answer", accept_answer)
graph.add_node("is_use", is_use)
graph.add_node("rewrite_question", rewrite_question)

graph.add_edge(START, "decide_retrieval")

graph.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {"retrieve": "retrieve", "generate_direct": "generate_direct"},
)

graph.add_edge("generate_direct", END)
graph.add_edge("retrieve", "is_relevant")

graph.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {"generate_from_context": "generate_from_context", "no_answer_found": "no_answer_found"},
)

graph.add_edge("no_answer_found", END)
graph.add_edge("generate_from_context", "is_sup")

graph.add_conditional_edges(
    "is_sup",
    route_after_issup,
    {"accept_answer": "accept_answer", "revise_answer": "revise_answer"},
)

graph.add_edge("revise_answer", "is_sup")   # loop back
graph.add_edge("accept_answer", "is_use")

graph.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {"END": END, "rewrite_question": "rewrite_question", "no_answer_found": "no_answer_found"},
)

graph.add_edge("rewrite_question", "retrieve")   # loop back

workflow = graph.compile()


# ---------------------------
# Run
# ---------------------------

initial_state = {
        "question": "What is a probabtion period?",
        "retrieval_query": "",
        "rewrite_tries": 0,
        "need_retrieval": False,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "",
        "evidence": [],
        "retries": 0,
        "isuse": "not_useful",
        "use_reason": "",
}

response = workflow.invoke(initial_state, config={"recursion_limit": 50})

print(response["answer"])
