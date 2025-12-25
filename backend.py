from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages import AIMessage
import os
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
client = InferenceClient()
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

from langgraph.graph.message import add_messages

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
  
def chat_node(state: ChatState):

    messages = state["messages"]

    hf_messages = []
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
        # ✅ KEEP HISTORY
        "messages": messages + [AIMessage(content=llm_response)]
    }
    
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = MemorySaver())