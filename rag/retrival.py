# rag/retrival.py

from .embedding import get_embedding
from .llm import generate_answer_with_context
from .vector_store import FaissVectorStore

vector_store = FaissVectorStore()

def rag_answer(question, tenant_id):
    query_embedding = get_embedding(question)
    results = vector_store.search(tenant_id, query_embedding, top_k=10)
    return generate_answer_with_context(question, results)
