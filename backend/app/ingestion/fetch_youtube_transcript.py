from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL
from uuid import uuid4
from app.embeddings import embed_model
from app.qdrant_store import qdrant_client, COLLECTION_NAME
from qdrant_client.models import PointStruct

def fetch_video_transcript(video_id: str, chat_id: str):
    snippets = YouTubeTranscriptApi().fetch(video_id)
    transcript = " ".join(s.text for s in snippets)
    
    # split text into chunks
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([transcript])
    texts = [d.page_content for d in docs]
    
    vectors = embed_model.embed_documents(texts)
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vectors[i],
            payload={
                "chat_id": chat_id,
                "page_content": texts[i],
                "metadata": {"source": "youtube", "video_id": video_id}
            }
        ) for i in range(len(texts))
    ]
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

def fetch_yt_title(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get("title")
