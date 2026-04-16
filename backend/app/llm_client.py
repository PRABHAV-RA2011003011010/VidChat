from huggingface_hub import InferenceClient
from app.config import HF_TOKEN, MODEL_ID

llm_client = InferenceClient(token=HF_TOKEN)

def get_completion(messages):
    completion = llm_client.chat.completions.create(
        model=MODEL_ID,
        messages=messages
    )
    return completion.choices[0].message.content