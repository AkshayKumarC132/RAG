import os
import tempfile
from typing import List
from pathlib import Path
import pytesseract
from PyPDF2 import PdfReader
from PIL import Image
from docx import Document
from pptx import Presentation
import pandas as pd
from project import settings

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import time

# Initialize Embeddings and Chat Model
embedding_model = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY,temperature=0)

# Qdrant initialization
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "xamplify_docs"


from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate


# Check if collection exists and create if it doesn't
def create_qdrant_collection_if_not_exists():
    try:
        # Check if the collection exists
        qdrant_client.get_collection(collection_name=collection_name)
        print(f"[+] Collection '{collection_name}' already exists.")
    except Exception:
        # Collection doesn't exist, create it
        print(f"[+] Collection '{collection_name}' not found. Creating it...")
        vector_params = VectorParams(size=1536, distance=Distance.COSINE)  # Adjust vector size and distance metric as needed
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
        )
        print(f"[+] Collection '{collection_name}' created successfully.")

create_qdrant_collection_if_not_exists()

# Qdrant
qdrant = QdrantVectorStore(
    client=QdrantClient(host="localhost", port=6333),
    collection_name="xamplify_docs",
    embedding=embedding_model,
)

# ----------- File Extraction Logic -----------

def extract_text_from_file(file_path: str, original_file_name: str) -> str:
    print(file_path)

    ext = Path(original_file_name).suffix.lower()  # <-- use original filename
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(file_path)
    elif ext == ".pptx":
        return extract_text_from_pptx(file_path)
    elif ext in [".xls", ".xlsx", ".csv"]:
        return extract_text_from_excel_or_csv(file_path)
    else:
        raise ValueError("Unsupported file type!")

def extract_text_from_pdf(path):
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)

def extract_text_from_pptx(path):
    prs = Presentation(path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def extract_text_from_excel(path):
    df = pd.read_excel(path)
    return df.to_string(index=False)

# def extract_text_from_excel_or_csv(path):
#     ext = Path(path).suffix.lower()
#     if ext == ".csv":
#         df = pd.read_csv(path)
#     else:
#         df = pd.read_excel(path)
#     return df.to_string(index=False)
def extract_text_from_excel_or_csv(path):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    elif ext == ".xls":
        df = pd.read_excel(path, engine="xlrd")
    else:
        raise ValueError("Unsupported Excel/CSV file extension!")
    return df.to_string(index=False)


def insert_document_to_vectorstore(text: str, source_type: str, file_ext: str):
    print("[+] Splitting text into smaller chunks...")
    docs = smart_split_text(text, file_ext)

    if source_type == "file":
        print(f"[+] Inserting {len(docs)} chunks into Qdrant... as file")
        qdrant.add_documents(docs,batch_size=64)
    elif source_type == "integration":
        print(f"[+] Inserting {len(docs)} chunks into Qdrant... as integration")
        qdrant.add_documents(docs,batch_size=64)
    else:
        raise ValueError("Invalid source_type")

    

def split_text_into_chunks(text: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([text])


from langchain.schema import Document as LangChainDocument  # <-- Rename it

def split_csv_rows(text: str, max_lines_per_chunk=50):
    lines = text.strip().split("\n")
    header = lines[0]
    data_lines = lines[1:]

    chunks = []
    for i in range(0, len(data_lines), max_lines_per_chunk):
        chunk_text = "\n".join([header] + data_lines[i:i+max_lines_per_chunk])
        chunks.append(LangChainDocument(page_content=chunk_text, metadata={"chunk": i // max_lines_per_chunk + 1}))
    return chunks

def split_unstructured_text(text: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents([text])
    for i, doc in enumerate(docs):
        doc.metadata["chunk"] = i + 1
    return docs

def smart_split_text(text: str, file_ext: str):
    if file_ext in [".csv", ".xls", ".xlsx"]:
        return split_csv_rows(text)
    else:
        return split_unstructured_text(text)



# ----------- Query Logic -----------

def ask_question(query: str, source_type: str):
    retriever = None
    start = time.time()
    print("Retrieving...")
    if source_type == "file":
        retriever = qdrant.as_retriever()
    elif source_type == "integration":
        retriever = qdrant.as_retriever()
    end = time.time()
    print("Retrieved in: ", end - start)

    print("Retriever: ", retriever)
    print("Query: ", query)

    start_time = time.time()
    print("Asking question...")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa.run(query)
    end_time = time.time()
    print("Question answered in: ", end_time - start_time)
    return result

# -----------  Deep Search + Reasoning ----------- 

# def ask_question(query: str, source_type: str):
#     retriever = None
#     if source_type == "file":
#         retriever = MultiQueryRetriever.from_llm(
#             retriever=qdrant.as_retriever(search_kwargs={"k": 20}),
#             llm=llm
#         )
#     elif source_type == "integration":
#         retriever = MultiQueryRetriever.from_llm(
#             retriever=qdrant.as_retriever(search_kwargs={"k": 20}),
#             llm=llm
#         )

#     reasoning_prompt = """
#     Use the following context to answer the question. Think step-by-step before concluding.
#     If you don't know the answer, say "I don't know" instead of making up facts.

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer (with reasoning):
#     """

#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type_kwargs={"prompt": PromptTemplate.from_template(reasoning_prompt)},
#         chain_type="stuff"  # or "map_reduce" for large inputs
#     )

#     result = qa.run(query)
#     return result

# def ask_question(query: str, source_type: str):
#     retriever = None
#     if source_type == "file":
#         retriever = qdrant.as_retriever()
#     elif source_type == "integration":
#         retriever = qdrant.as_retriever()

#     # 1. Define a smarter prompt
#     prompt_template = """You are a helpful assistant. 
#         Use the following context to answer the question.
#         If unsure, reason carefully and answer based on the information provided.

#         Context:
#         {context}

#         Question:
#         {question}

#         Answer (step-by-step reasoning first, then final answer):
#         """

#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=prompt_template,
#     )

#     # 2. Create a better QA Chain with reasoning prompt
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",  # you can try "map_reduce" or "refine" for even deeper reasoning
#         chain_type_kwargs={"prompt": prompt},
#     )

#     # 3. Run the query
#     result = qa_chain.run(query)
#     return result
