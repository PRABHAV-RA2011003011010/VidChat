import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
COLLECTION_NAME = "rag_docs"

if not all([HF_TOKEN, QDRANT_URL, QDRANT_API_KEY]):
    raise RuntimeError("Missing required environment variables")
