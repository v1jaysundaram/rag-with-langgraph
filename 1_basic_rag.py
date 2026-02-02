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
    context: List[Document]
    augmented_query: str
    answer: str

# prompt
prompt = PromptTemplate(
        template="""
        You are an expert chat assistant designed to answer questions based on provided context.
        Carefully use the context below to answer the question. 
        If the context does not contain enough information, respond clearly that you do not know the answer.

        Question: {question}
        Context: {context}
    
        Answer:""",
        input_variables=['context', 'question']
    )

# nodes
def retrieve(state: RagState):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(state["query"])
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return {"context": docs_content}


def augment(state: RagState):    
    augmented_query = prompt.invoke({"question": state["query"], "context": state["context"]})
    return {"augmented_query": augmented_query}


def generate(state: RagState):  
    response = llm.invoke(state["augmented_query"])
    return {"answer": response.content}

# define graph
graph = StateGraph(RagState)

graph.add_node("retrieve", retrieve)
graph.add_node("augment", augment)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "augment")
graph.add_edge("augment", "generate")
graph.add_edge("generate", END)

workflow = graph.compile()


# test run
input_state = {"query": "What is a RAG and what are its components?"}

response = workflow.invoke(input_state)
print(response["answer"])