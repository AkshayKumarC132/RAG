from openai import OpenAI
from project import settings
import os

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding
