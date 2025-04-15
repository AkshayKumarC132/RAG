# rag/vector_store.py

from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

class QdrantVectorStore:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)

    def upsert(self, tenant_id, documents_with_embeddings):
        collection = f"{tenant_id}_collection"
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=doc["embedding"],
                payload={"text": doc["text"], "tenant_id": tenant_id}
            )
            for doc in documents_with_embeddings
        ]
        self.client.upsert(collection_name=collection, points=points)

    def search(self, tenant_id, query_embedding, top_k=5):
        collection = f"{tenant_id}_collection"
        results = self.client.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=top_k
        )
        return [res.payload["text"] for res in results]
