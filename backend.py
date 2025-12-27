from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages import AIMessage
from langgraph.graph.message import add_messages
import os
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional
from langchain.docstore import InMemoryDocstore

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
client = InferenceClient()
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)

# 4️⃣ Empty docstore
docstore = InMemoryDocstore({})

# 5️⃣ Create FAISS vector store
vector_store = FAISS(embedding_function=embed_model, index=index, docstore=docstore)


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    context: Optional[str]

def route_after_start(state: ChatState):
    if vector_store.index.ntotal > 0:
        return "retrieval_node"
    return "chat_node"

def retrieval_node(state: ChatState):
    messages = state["messages"]

    # latest user query
    query = next(
        msg.content for msg in reversed(messages)
        if isinstance(msg, HumanMessage)
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    retrieved_docs = retriever.invoke(query)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    return {
        "context": context_text
    }
  
def chat_node(state: ChatState):

    messages = state["messages"]
    context = state.get("context", "")

    hf_messages = []
    
    if context:
        hf_messages.append({
            "role": "system",
            "content": f"Use the following context to answer:\n\n{context}"
        })    
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            hf_messages.append({
                "role": "user",
                "content": msg.content
            })
        elif isinstance(msg, AIMessage):
            hf_messages.append({
                "role": "assistant",
                "content": msg.content
            })

    # ✅ CORRECT CALL
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=hf_messages
    )

    llm_response = completion.choices[0].message.content

    return {
        
        "messages": messages + [AIMessage(content=llm_response)]
    }
    
graph = StateGraph(ChatState)

graph.add_node("retrieval_node", retrieval_node)
graph.add_node("chat_node", chat_node)

graph.add_conditional_edges(
    START,
    route_after_start,
    {
        "retrieval_node": "retrieval_node",
        "chat_node": "chat_node",
    }
)

graph.add_edge("retrieval_node", "chat_node")
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = MemorySaver())



def fetch_video_transcript(video_id: str) -> str:
    
    ytt_api = YouTubeTranscriptApi()
    fetched_snippets = ytt_api.fetch(video_id)

    transcript=[]
    for snippet in fetched_snippets:
        transcript.append(snippet.text)
    transcript=" ".join(transcript)
    
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([transcript])

    vector_store.add_documents(chunks)
    
    return transcript

