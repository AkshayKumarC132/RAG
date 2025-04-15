# rag/vector_store.py

# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# import uuid

# class QdrantVectorStore:
#     def __init__(self, host="localhost", port=6333):
#         self.client = QdrantClient(host=host, port=port)

#     def upsert(self, tenant_id, documents_with_embeddings):
#         collection = f"{tenant_id}_collection"
#         self.client.recreate_collection(
#             collection_name=collection,
#             vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
#         )

#         points = [
#             models.PointStruct(
#                 id=str(uuid.uuid4()),
#                 vector=doc["embedding"],
#                 payload={"text": doc["text"], "tenant_id": tenant_id}
#             )
#             for doc in documents_with_embeddings
#         ]
#         self.client.upsert(collection_name=collection, points=points)

#     def search(self, tenant_id, query_embedding, top_k=5):
#         collection = f"{tenant_id}_collection"
#         results = self.client.search(
#             collection_name=collection,
#             query_vector=query_embedding,
#             limit=top_k
#         )
#         return [res.payload["text"] for res in results]

# from pinecone import Pinecone, ServerlessSpec
# import uuid
# from project import settings

# class PineconeVectorStore:
#     def __init__(self, environment="us-west1-gcp"):
#         self.client = Pinecone(api_key=settings.OPENAI_API_KEY)
#         self.environment = environment

#     def upsert(self, tenant_id, documents_with_embeddings):
#         index_name = f"{tenant_id}-collection"
        
#         # Create index if it doesn't exist
#         if index_name not in self.client.list_indexes().names():
#             self.client.create_index(
#                 name=index_name,
#                 dimension=1536,  # Match embedding size
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud="aws", region="us-west-2")
#             )

#         # Connect to index
#         index = self.client.Index(index_name)

#         # Prepare vectors
#         vectors = [
#             {
#                 "id": str(uuid.uuid4()),
#                 "values": doc["embedding"],
#                 "metadata": {"text": doc["text"], "tenant_id": tenant_id}
#             }
#             for doc in documents_with_embeddings
#         ]

#         # Upsert vectors
#         index.upsert(vectors=vectors)

#     def search(self, tenant_id, query_embedding, top_k=5):
#         index_name = f"{tenant_id}-collection"
#         index = self.client.Index(index_name)
#         results = index.query(
#             vector=query_embedding,
#             top_k=top_k,
#             include_metadata=True
#         )
#         return [match["metadata"]["text"] for match in results["matches"]]


import faiss
import numpy as np
import pickle
import os
import uuid

class FaissVectorStore:
    def __init__(self, storage_path="faiss_index"):
        self.storage_path = storage_path
        self.index = None
        self.metadata = {}
        self.dimension = 1536  # Match embedding size

    def upsert(self, tenant_id, documents_with_embeddings):
        # Initialize index
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # HNSW with M=32
        self.index.hnsw.efConstruction = 100
        self.metadata = {}

        # Prepare vectors and metadata
        vectors = np.array([doc["embedding"] for doc in documents_with_embeddings], dtype=np.float32)
        for i, doc in enumerate(documents_with_embeddings):
            doc_id = str(uuid.uuid4())
            self.metadata[doc_id] = {"text": doc["text"], "tenant_id": tenant_id}

        # Add vectors to index
        self.index.add(vectors)

        # Save index and metadata
        faiss.write_index(self.index, f"{self.storage_path}_{tenant_id}.index")
        with open(f"{self.storage_path}_{tenant_id}.metadata", "wb") as f:
            pickle.dump(self.metadata, f)

    def search(self, tenant_id, query_embedding, top_k=5):
        # Load index and metadata if not in memory
        if self.index is None:
            self.index = faiss.read_index(f"{self.storage_path}_{tenant_id}.index")
            with open(f"{self.storage_path}_{tenant_id}.metadata", "rb") as f:
                self.metadata = pickle.load(f)

        # Search
        query = np.array([query_embedding], dtype=np.float32)
        _, indices = self.index.search(query, top_k)
        results = []
        for idx in indices[0]:
            doc_id = list(self.metadata.keys())[idx]
            results.append(self.metadata[doc_id]["text"])
        return results