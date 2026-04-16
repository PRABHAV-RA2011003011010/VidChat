from typing import TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    chat_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    context: Optional[str]