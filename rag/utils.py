from pathlib import Path
import pytesseract
from PyPDF2 import PdfReader
from PIL import Image
from docx import Document
from pptx import Presentation
import pandas as pd
from project import settings
from pdfminer.high_level import extract_text as pdfminer_extract_text
import cv2
import pandas as pd
import spacy
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import time
from sentence_transformers import SentenceTransformer
from langchain.schema import Document as LangChainDocument  # <-- Rename it
import json
import xml.etree.ElementTree as ET

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from langchain.chains.llm import LLMChain

# model = SentenceTransformer('all-MiniLM-L6-v2')

nlp = spacy.load("en_core_web_sm")  # For NER

# Initialize Embeddings and Chat Model
embedding_model = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY,temperature=0)

# Qdrant initialization
qdrant_client = QdrantClient(host="localhost", port=6333)
collection_name = "xamplify_docs"




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
        # vector_params = VectorParams(size=384, distance=Distance.COSINE)  # 384 for MiniLM
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
        )
        print(f"[+] Collection '{collection_name}' created successfully.")
        
# qdrant_client.delete_collection(collection_name="xamplify_docs")

create_qdrant_collection_if_not_exists()

# from langchain_community.embeddings import HuggingFaceEmbeddings

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# Qdrant
qdrant = QdrantVectorStore(
    client=QdrantClient(host="localhost", port=6333),
    collection_name="xamplify_docs",
    embedding=embedding_model,
)

# ----------- File Extraction Logic -----------

def extract_text_from_file(file_path: str, original_file_name: str) -> str:

    ext = Path(original_file_name).suffix.lower()
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
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".json":
        return extract_text_from_json(file_path)
    elif ext == ".xml":
        return extract_text_from_xml(file_path)
    elif ext == ".html":
        return extract_text_from_html(file_path)
    elif ext == ".md":  # Adding support for Markdown files
        return extract_text_from_md(file_path)
    elif ext == ".yaml" or ext == ".yml":  # Adding support for YAML files
        return extract_text_from_yaml(file_path)
    elif ext in [".ini", ".cfg"]:  # Adding support for configuration files
        return extract_text_from_ini(file_path)
    else:
        raise ValueError("Unsupported file type!")

# ----------- Dynamic Metadata Extraction Logic -----------

def extract_metadata(text: str, file_ext: str) -> dict:
    metadata = {}

    if file_ext in [".csv", ".xls", ".xlsx"]:
        try:
            # Try extracting header columns
            lines = text.strip().split("\n")
            header = lines[0].split()
            for col in header:
                metadata[f"column_{col.lower()}"] = True
        except Exception as e:
            print(f"[!] CSV/Excel metadata extraction failed: {e}")
    
    elif file_ext in [".pdf", ".docx", ".pptx", ".txt"]:
        doc = nlp(text[:3000])  # Analyze only first 3000 chars for speed
        entities = {"ORG": [], "GPE": [], "DATE": []}
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        if entities["ORG"]:
            metadata["organizations"] = list(set(entities["ORG"]))
        if entities["GPE"]:
            metadata["locations"] = list(set(entities["GPE"]))
        if entities["DATE"]:
            metadata["dates"] = list(set(entities["DATE"]))

    elif file_ext in [".png", ".jpg", ".jpeg"]:
        metadata["detected_from"] = "image"
    
    else:
        metadata["type"] = "unknown"

    return metadata

# ----------- Specific Extractors -----------

# 1. PDF
def extract_text_from_pdf(path):
    try:
        text = pdfminer_extract_text(path)
    except Exception as e:
        print(f"[!] PDFMiner failed: {e}")
        text = ""
    return text.strip()

# 2. DOCX
def extract_text_from_docx(path):
    doc = Document(path)
    fullText = []

    # Extract paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            fullText.append(para.text)

    # Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            fullText.append('\t'.join(row_data))

    return '\n'.join(fullText).strip()

# 3. PPTX
def extract_text_from_pptx(path):
    prs = Presentation(path)
    text = ""

    for slide in prs.slides:
        # Slide shapes
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text += shape.text + "\n"
        # Slide notes
        if slide.has_notes_slide:
            notes_slide = slide.notes_slide
            if notes_slide and notes_slide.notes_text_frame:
                text += notes_slide.notes_text_frame.text + "\n"

    return text.strip()

# 4. Excel/CSV
def extract_text_from_excel_or_csv(path):
    ext = Path(path).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".xlsx":
        xls = pd.ExcelFile(path, engine="openpyxl")
        dfs = [xls.parse(sheet_name) for sheet_name in xls.sheet_names]
        df = pd.concat(dfs, keys=xls.sheet_names)
    elif ext == ".xls":
        xls = pd.ExcelFile(path, engine="xlrd")
        dfs = [xls.parse(sheet_name) for sheet_name in xls.sheet_names]
        df = pd.concat(dfs, keys=xls.sheet_names)
    else:
        raise ValueError("Unsupported Excel/CSV file extension!")

    # Convert dataframe to markdown-like format
    return df.to_string(index=False)

# 5. Image
def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def extract_text_from_image(path):
    preprocessed_img = preprocess_image(path)
    text = pytesseract.image_to_string(preprocessed_img)
    return text.strip()

# 6. TXT
def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 7. JSON
def extract_text_from_json(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=4)  # pretty prints JSON

# 8. XML
def extract_text_from_xml(file_path: str) -> str:
    tree = ET.parse(file_path)
    root = tree.getroot()
    return ET.tostring(root, encoding='utf-8').decode('utf-8')

# 9. HTML
def extract_text_from_html(file_path: str) -> str:
    from bs4 import BeautifulSoup
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text()

# 10. Markdown
def extract_text_from_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 11. YAML
def extract_text_from_yaml(file_path: str) -> str:
    import yaml
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return yaml.dump(data)

# 12. INI
def extract_text_from_ini(file_path: str) -> str:
    import configparser
    config = configparser.ConfigParser()
    config.read(file_path)
    output = []
    for section in config.sections():
        output.append(f"[{section}]")
        for key, value in config.items(section):
            output.append(f"{key} = {value}")
    return "\n".join(output)

# ----------- VectorStore Insert Logic -----------

def insert_document_to_vectorstore(text: str, source_type: str, file_ext: str,vector_id: str):
    
    docs = smart_split_text(text, file_ext)

    # Extract global metadata once for the whole file
    global_metadata = extract_metadata(text, file_ext)
    global_metadata["vector_id"] = vector_id  # Add vector_id to metadata


    # Merge the global metadata into each document's metadata
    for doc in docs:
        if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
            doc.metadata = {}
        doc.metadata.update(global_metadata)


    if source_type == "file":
        print(f"[+] Inserting {len(docs)} chunks into Qdrant... as file")
        qdrant.add_documents(docs, batch_size=64)
    elif source_type == "integration":
        print(f"[+] Inserting {len(docs)} chunks into Qdrant... as integration")
        qdrant.add_documents(docs, batch_size=64)
    else:
        raise ValueError("Invalid source_type")

    

def split_text_into_chunks(text: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([text])



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

# def ask_question(query: str, source_type: str):
#     retriever = None
#     start = time.time()

#     if source_type == "file":
#         retriever = qdrant.as_retriever(search_kwargs={"k": 5})
#         # retriever = qdrant.as_retriever()
#     elif source_type == "integration":
#         retriever = qdrant.as_retriever(search_kwargs={"k": 5})
#     end = time.time()
#     print("Retrieved in: ", end - start)

    # start_time = time.time()
    # qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    # result = qa.run(query)
    # end_time = time.time()
    # print("Question answered in: ", end_time - start_time)
    # return result

# def ask_question(query: str, source_type: str, documents=None):
#     from langchain.chains.retrieval_qa.base import RetrievalQA
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     from langchain.schema import Document as LangChainDocument
    
#     retriever = None
#     start = time.time()

#     if documents:
#         # Use provided documents as context
#         text = "\n".join(doc["content"] for doc in documents)
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         docs = splitter.create_documents([text])
#         for i, doc in enumerate(docs):
#             doc.metadata["chunk"] = i + 1
#         retriever = qdrant.as_retriever(search_kwargs={"k": 5})
#     else:
#         # Default behavior: retrieve from entire collection
#         retriever = qdrant.as_retriever(search_kwargs={"k": 5})

#     end = time.time()
#     print("Retrieved in: ", end - start)

#     start_time = time.time()
#     qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
#     result = qa.run(query)
#     end_time = time.time()
#     print("Question answered in: ", end_time - start_time)
#     return result

# ========================Previous Working Code==========================
# def ask_question(query: str, source_type: str, documents=None):
#     try:
#         start = time.time()

#         # Define the default system prompt
#         system_prompt = """
#         You are a helpful document analyst. Your role is to provide accurate, concise, and detailed answers based on the provided document context. 
#         Use the information in the documents to answer the question directly and avoid including irrelevant details. 
#         If the documents do not contain enough information to answer the question, state so clearly.
#         Format your response clearly and professionally, using bullet points or paragraphs as appropriate.
#         """

#         # Create a prompt template for the QA chain
#         prompt_template = PromptTemplate(
#             template="""{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:""",
#             input_variables=["system_prompt", "context", "question"]
#         )

#         if documents:
#             print("[+] Using provided documents as context")
#             # Create a filter for the vector_id
#             vector_id = documents[0]["metadata"]["vector_id"]
#             filter = Filter(
#                 must=[
#                     FieldCondition(
#                         key="metadata.vector_id",
#                         match=MatchValue(value=vector_id)
#                     )
#                 ]
#             )
#             # Use the existing Qdrant collection with a filter
#             print(len(documents))
#             print("[+] Filter for vector_id:", vector_id)
#             print("[+] Filter:", filter)
#             retriever = qdrant.as_retriever(search_kwargs={"k": len(documents), "filter": filter})
#             print(f"[+] Created retriever with filter for vector_id: {vector_id}")
#         else:
#             # Use the default Qdrant collection
#             retriever = qdrant.as_retriever(search_kwargs={"k": 5})
#             print("[+] Using default Qdrant collection retriever")

#         end = time.time()
#         print(f"[+] Retrieved in: {end - start:.2f} seconds")

#         start_time = time.time()
#         # Configure the RetrievalQA chain with the custom prompt
#         qa = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type_kwargs={
#                 "prompt": prompt_template.partial(system_prompt=system_prompt)
#             }
#         )
#         # Run the query with increased max_tokens
#         result = qa.invoke({"query": query})["result"]  # Use invoke for modern LangChain versions
#         end_time = time.time()
#         print(f"[+] Question answered in: {end_time - start_time:.2f} seconds")

#         return result

#     except Exception as e:
#         print(f"[!] Error in ask_question: {e}")
#         raise



def summarize_batch(batch_docs):
    combined_context = "\n".join([doc.page_content for doc in batch_docs])
    summary = summarize_context(combined_context)
    
    # Make sure it's a string
    return summary if isinstance(summary, str) else str(summary)

def ask_question(query: str, source_type: str, documents=None):
    try:
        start = time.time()

        system_prompt = """
        You are a helpful document analyst. Your role is to provide accurate, concise, and detailed answers based on the provided document context. 
        Use the information in the documents to answer the question directly and avoid including irrelevant details. 
        If the documents do not contain enough information to answer the question, state so clearly.
        """

        final_context = ""

        if documents:
            print("[+] Using provided documents as context")
            vector_id = documents[0]["metadata"]["vector_id"]
            filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.vector_id",
                        match=MatchValue(value=vector_id)
                    )
                ]
            )
            retriever = qdrant.as_retriever(search_kwargs={"k": len(documents), "filter": filter})
            retrieved_docs = retriever.get_relevant_documents(query)
            print(f"[+] Retrieved {len(retrieved_docs)} documents")

            # Map Phase: Summarize batches in parallel
            batch_size = 5
            batches = list(batch_documents(retrieved_docs, batch_size=batch_size))

            summaries = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(summarize_batch, batch) for batch in batches]
                for future in futures:
                    summaries.append(future.result())

            # Reduce Phase: Summarize all summaries into one
            combined_summary = "\n".join(summaries)
            final_context = summarize_context(combined_summary)

            print(f"[+] Summarized {len(batches)} batches and reduced to final context")

        else:
            print("[+] Using default Qdrant collection retriever")
            retriever = qdrant.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.get_relevant_documents(query)
            final_context = "\n".join([doc.page_content for doc in retrieved_docs])

        end = time.time()
        print(f"[+] Retrieved and processed context in: {end - start:.2f} seconds")

        # Final QA
        prompt_template = PromptTemplate(
            template="""{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:""",
            input_variables=["system_prompt", "context", "question"]
        )

        qa_chain = LLMChain(
            llm=llm,
            prompt=prompt_template.partial(system_prompt=system_prompt, context=final_context)
        )

        start_time = time.time()
        result = qa_chain.invoke({"question": query})["text"]
        end_time = time.time()
        print(f"[+] Question answered in: {end_time - start_time:.2f} seconds")

        return result

    except Exception as e:
        print(f"[!] Error in ask_question: {e}")
        raise


def summarize_context(context_text):
    """Summarizes a large text into a small focused version."""
    prompt = f"Summarize the following content concisely and accurately:\n\n{context_text}\n\nSummary:"
    
    response = llm.invoke(prompt)
    
    # Handle if response is AIMessage, dict, or just text
    if hasattr(response, "content"):
        return response.content
    elif isinstance(response, dict) and "text" in response:
        return response["text"]
    else:
        return str(response)


def batch_documents(documents, batch_size=5):
    """Yield successive batches of documents."""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]


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



def retrieve_documents_by_vector_id(vector_id: str) -> list:
    try:
        # Create a filter for the vector_id
        filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.vector_id",
                    match=MatchValue(value=vector_id)
                )
            ]
        )

        print("[+] Searching for documents with vector_id:", vector_id)

        # Query Qdrant for documents matching the vector_id
        search_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=filter,
            limit=100,  # Adjust based on expected number of chunks
            with_vectors=False,
            with_payload=True
        )

        print("[+] Search result:", search_result)
        if not search_result[0]:  # Check if any records were found
            print("[!] No documents found for the given vector_id.")
            return []

        # Extract documents from the search result
        documents = []
        for record in search_result[0]:  # search_result[0] contains the records
            doc_content = record.payload.get("page_content", "")
            doc_metadata = record.payload.get("metadata", {})
            documents.append({
                "content": doc_content,
                "metadata": doc_metadata
            })

        return documents

    except Exception as e:
        print(f"[!] Error retrieving documents by vector_id: {e}")
        return []
    
def delete_documents_by_vector_id(vector_id: str) -> list:
    try:
        # Delete from Qdrant
        print(f"[+] Deleting documents from Qdrant with vector_id: {vector_id}")
        filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.vector_id",  # Adjust based on Qdrant payload structure
                    match=MatchValue(value=vector_id)
                )
            ]
        )
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=filter
        )
        print(f"[+] Successfully deleted Qdrant documents with vector_id: {vector_id}")
        return True
    except Exception as e:
        print(f"[!] Error deleting documents from Qdrant: {e}")
        return False