from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from app.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
from app.embeddings import embed_model

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

def get_vector_store(chat_id: str) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embed_model,
        content_payload_key="page_content",
        metadata_payload_key="metadata",
    )
