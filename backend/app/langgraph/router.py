from app.langgraph.state import ChatState
from app.qdrant_store import qdrant_client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue

def route_after_start(state: ChatState):
    result, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1,
        with_payload=False,
        scroll_filter=Filter(
            must=[FieldCondition(key="chat_id", match=MatchValue(value=state["chat_id"]))]
        )
    )
    if len(result) > 0:
        return "retrieval_node"
    return "chat_node"
