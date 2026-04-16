from app.langgraph.state import ChatState
from app.qdrant_store import get_vector_store
from langchain_core.messages import HumanMessage, AIMessage
from app.llm_client import get_completion
from qdrant_client.models import Filter, FieldCondition, MatchValue

def retrieval_node(state: ChatState):
    vector_store = get_vector_store(state["chat_id"])
    messages = state["messages"]

    query = next(msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage))

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": Filter(
                must=[FieldCondition(key="chat_id", match=MatchValue(value=state["chat_id"]))]
            )
        }
    )

    retrieved_docs = retriever.invoke(query)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    return {"context": context_text}

def chat_node(state: ChatState):
    messages = state["messages"]
    context = state.get("context", "")
    hf_messages = []

    if context:
        hf_messages.append({"role": "system", "content": f"Use the following context:\n{context}"})

    for msg in messages:
        if isinstance(msg, HumanMessage):
            hf_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            hf_messages.append({"role": "assistant", "content": msg.content})

    llm_response = get_completion(hf_messages)
    return {"messages": messages + [AIMessage(content=llm_response)]}
