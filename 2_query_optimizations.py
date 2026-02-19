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

persist_path="rag_embeddings"

vector_store = FAISS.load_local(
    folder_path=persist_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
    ) 

# langgraph app

llm = ChatOpenAI(model_name = 'gpt-4o')


# state
class RagState(TypedDict):
    query: str
    typofree_query: str
    optimized_query: str
    docs: List[Document]
    context: str
    augmented_query: str
    answer: str

# prompts
prompt_rag = PromptTemplate(
        template="""
        You are an expert chat assistant designed to answer questions based on provided context.
        Carefully use the context below to answer the question. 
        If the context does not contain enough information, respond clearly that you do not know the answer.

        Question: {query}
        Context: {context}""",

        input_variables=['context', 'query']
    )

prompt_typo = PromptTemplate(
        template="""You are an expert AI assistant tasked with rewriting user queries to improve retrieval in a RAG system. 
Given an original query, remove any typos and rewrite it to be free from any spelling mistakes.

Original Query: {query}""",

        input_variables=['query']
) 


prompt_optimization = PromptTemplate(
        template="""You are an expert AI assistant tasked with optimizing user queries to improve retrieval in a RAG system. 
Given a typo free query, generate a query that is broad and general, which can help retrieve relevant background information.

Typo Free Query: {typofree_query}""",

        input_variables=['typofree_query']
)


# nodes

def typo_correction(state: RagState):
    prompt_text = prompt_typo.format(query=state["query"])
    response = llm.invoke(prompt_text)
    return {"typofree_query": response.content}

def optimize_query(state: RagState):
    prompt_text = prompt_optimization.format(typofree_query=state["typofree_query"])
    response = llm.invoke(prompt_text)
    return {"optimized_query": response.content}

def retrieve(state: RagState):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(state["optimized_query"])
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return {"docs": retrieved_docs, "context": docs_content}


def augment(state: RagState):
    augmented_query = prompt_rag.format(query=state["query"], context=state["context"])
    return {"augmented_query": augmented_query}


def generate(state: RagState):  
    response = llm.invoke(state["augmented_query"])
    return {"answer": response.content}

# define graph
graph = StateGraph(RagState)

graph.add_node("typo_correction", typo_correction)
graph.add_node("optimize_query", optimize_query)
graph.add_node("retrieve", retrieve)
graph.add_node("augment", augment)
graph.add_node("generate", generate)

graph.add_edge(START, "typo_correction")
graph.add_edge("typo_correction", "optimize_query")
graph.add_edge("optimize_query", "retrieve")
graph.add_edge("retrieve", "augment")
graph.add_edge("augment", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()


# test run
input_state = {"query": "What is a LLM and what is its compoents?"}

response = workflow.invoke(input_state)
print(response["answer"])