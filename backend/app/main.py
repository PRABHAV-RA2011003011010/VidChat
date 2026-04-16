from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.langgraph.state import ChatState
from app.langgraph.router import route_after_start
from app.langgraph.nodes import retrieval_node, chat_node

graph = StateGraph(ChatState)
graph.add_node("retrieval_node", retrieval_node)
graph.add_node("chat_node", chat_node)
graph.add_conditional_edges(START, route_after_start, {"retrieval_node": "retrieval_node", "chat_node": "chat_node"})
graph.add_edge("retrieval_node", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=MemorySaver())
