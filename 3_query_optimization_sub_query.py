from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import json

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
    subqueries: List[str]
    docs: List[Document]
    context: str
    augmented_query: str
    answer: str

# prompt
prompt_rag = PromptTemplate(
        template="""
        You are an expert chat assistant designed to answer questions based on provided context.
        Carefully use the context below to answer the question. 
        If the context does not contain enough information, respond clearly that you do not know the answer.

        Question: {query}
        Context: {context}""",

        input_variables=['context', 'query']
    )

prompt_subquery = PromptTemplate(
        template="""You are an expert AI assistant tasked with optimizing user queries to improve retrieval in a RAG system.
Given a query, break it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the query. The answer should be strictly be a valid JSON list of strings.

Query: {query}

Example: How can I stay healthy?

Answer:
[What should I eat to stay healthy?, How often should I exercise to stay healthy?, How can I manage stress to stay healthy?, How important is sleep for staying healthy?]
""",


        input_variables=['query'])

# nodes
def subquery(state: RagState):
    response = llm.invoke(prompt_subquery.format(query=state["query"]))
    
    subqueries = json.loads(response.content.strip())
    
    return {"subqueries": subqueries}



def retrieve(state: RagState):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    unique_docs = {}
    
    for q in state["subqueries"]:
        docs = retriever.invoke(q)
        for d in docs:
            unique_docs[d.page_content] = d  # dedupe by content

    merged_docs = list(unique_docs.values())
    context = "\n\n".join(doc.page_content for doc in merged_docs)

    return {"docs": merged_docs, "context": context}


def augment(state: RagState):
    augmented_query = prompt_rag.format(query=state["query"], context=state["context"])
    return {"augmented_query": augmented_query}

def generate(state: RagState):  
    response = llm.invoke(state["augmented_query"])
    return {"answer": response.content}

# define graph
graph = StateGraph(RagState)

graph.add_node("subquery", subquery)
graph.add_node("retrieve", retrieve)
graph.add_node("augment", augment)
graph.add_node("generate", generate)

graph.add_edge(START, "subquery")
graph.add_edge("subquery", "retrieve")
graph.add_edge("retrieve", "augment")
graph.add_edge("augment", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()


# test run
input_state = {"query": "What is a LLM, and what are the components?"}

response = workflow.invoke(input_state)
print(response["answer"])