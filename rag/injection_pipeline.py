# rag/injection_pipeline.py

from .embedding import get_embedding
from .vector_store import QdrantVectorStore
import textwrap

vector_store = QdrantVectorStore()

def ingest_document(text, tenant_id, title="Uploaded Doc"):
    chunks = textwrap.wrap(text, 500)
    documents = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        documents.append({"text": chunk, "embedding": embedding})
    print(f"Upserting {len(documents)} chunks for tenant {tenant_id}...")
    vector_store.upsert(tenant_id, documents)
