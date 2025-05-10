#  rag/utils.py

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
import spacy
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, MatchValue, Filter, FieldCondition
import time
from langchain.schema import Document as LangChainDocument
import json
import xml.etree.ElementTree as ET
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from langchain.chains.llm import LLMChain
from .models import Document as DocumentModel, OpenAIKey, DocumentAlert
import os
import subprocess
import platform
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from moviepy.editor import VideoFileClip
import whisper
import requests
from urllib.parse import urlparse
from knox.models import AuthToken
from django.shortcuts import get_object_or_404

nlp = spacy.load("en_core_web_sm")

def get_authenticated_user(token: str):
    auth_token = get_object_or_404(AuthToken, token_key=token)
    return auth_token.user

def get_openai_api_key(user):
    openai_key = OpenAIKey.objects.filter(user=user, is_valid=True).first()
    return openai_key.api_key if openai_key else settings.OPENAI_API_KEY

def initialize_qdrant_collection(collection_name):
    qdrant_client = QdrantClient(host="localhost", port=6333)
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        print(f"[+] Collection '{collection_name}' already exists.")
    except Exception:
        print(f"[+] Creating collection '{collection_name}'...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        print(f"[+] Collection '{collection_name}' created successfully.")
    return qdrant_client

def get_qdrant_vector_store(user, collection_name):
    api_key = get_openai_api_key(user)
    embedding_model = OpenAIEmbeddings(api_key=api_key)
    qdrant_client = initialize_qdrant_collection(collection_name)
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

def process_file(file=None, s3_file_url=None, file_name=None):
    if file:
        file_name = file_name or file.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
            for chunk in file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
    elif s3_file_url:
        parsed_url = urlparse(s3_file_url)
        file_name = file_name or parsed_url.path.split("/")[-1]
        response = requests.get(s3_file_url)
        if response.status_code != 200:
            raise ValueError("Failed to download file from S3 URL.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
    else:
        raise ValueError("Either file or s3_file_url must be provided.")
    return tmp_path, file_name

def extract_text_from_file(file_path: str, original_file_name: str) -> str:
    ext = Path(original_file_name).suffix.lower()
    extracted_data = None
    print(f"[+] Starting text extraction for {original_file_name} with extension {ext}...")
    
    if ext == ".pdf":
        extracted_data = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        extracted_data = extract_text_from_docx(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        extracted_data = extract_text_from_image(file_path)
    elif ext == ".pptx":
        extracted_data = extract_text_from_pptx(file_path)
    elif ext in [".xls", ".xlsx", ".csv"]:
        extracted_data = extract_structured_from_excel_or_csv(file_path)
        if isinstance(extracted_data, dict):
            extracted_data = json.dumps(extracted_data)
        elif isinstance(extracted_data, list):
            extracted_data = '\n'.join([json.dumps(item) for item in extracted_data])
    elif ext == ".txt":
        extracted_data = extract_text_from_txt(file_path)
    elif ext == ".json":
        extracted_data = extract_structured_from_json(file_path)
        if isinstance(extracted_data, dict):
            extracted_data = json.dumps(extracted_data)
    elif ext == ".xml":
        extracted_data = extract_text_from_xml(file_path)
    elif ext == ".html":
        extracted_data = extract_text_from_html(file_path)
    elif ext == ".md":
        extracted_data = extract_text_from_md(file_path)
    elif ext in [".yaml", ".yml"]:
        extracted_data = extract_text_from_yaml(file_path)
    elif ext in [".ini", ".cfg"]:
        extracted_data = extract_text_from_ini(file_path)
    elif ext == ".ppt":
        extracted_data = extract_text_from_ppt(file_path)
    elif ext in [".mp4", ".avi", ".mov"]:
        extracted_data = extract_text_from_video(file_path)
    elif ext in [".mp3", ".wav", ".m4a"]:
        extracted_data = extract_text_from_audio(file_path)
    else:
        raise ValueError("Unsupported file type!")

    print(f"[+] Successfully extracted data from {original_file_name}: {type(extracted_data)}")
    return extracted_data

def extract_text_from_video(file_path: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            video = VideoFileClip(file_path)
            audio = video.audio
            audio.write_audiofile(tmp_audio.name, logger=None)
            return extract_text_from_audio(tmp_audio.name)
    except Exception as e:
        print(f"[!] Error processing video: {e}")
        return ""

def extract_text_from_audio(file_path: str) -> str:
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path, language="en")
        return result["text"]
    except Exception as e:
        print(f"[!] Whisper failed: {e}")
        return ""

def extract_metadata(data, file_ext: str) -> dict:
    metadata = {}
    if file_ext in [".csv", ".xls", ".xlsx", ".json"]:
        try:
            if isinstance(data, (dict, list)):
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    header = data[0].keys()
                elif isinstance(data, dict):
                    header = data.keys()
                else:
                    header = []
                for col in header:
                    metadata[f"column_{col.lower()}"] = True
        except Exception as e:
            print(f"[!] Structured metadata extraction failed: {e}")
    elif file_ext in [".pdf", ".docx", ".pptx", ".txt"]:
        if isinstance(data, str):
            doc = nlp(data[:3000])
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

def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("The uploaded PDF is password-protected. Please upload an unlocked PDF.")
        text = pdfminer_extract_text(path)
        if not text.strip():
            raise ValueError("The PDF appears to be empty or unreadable after extraction.")
    except ValueError as ve:
        print(f"[!] PDF Extraction error: {ve}")
        raise ve
    except Exception as e:
        print(f"[!] Unexpected error during PDF extraction: {e}")
        raise ValueError("Failed to extract text from PDF.") from e
    return text.strip()

def extract_text_from_docx(path):
    doc = Document(path)
    fullText = []
    for para in doc.paragraphs:
        if para.text.strip():
            fullText.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            fullText.append('\t'.join(row_data))
    return '\n'.join(fullText).strip()

def extract_text_from_pptx(path):
    prs = Presentation(path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text += shape.text + "\n"
        if slide.has_notes_slide:
            notes_slide = slide.notes_slide
            if notes_slide and notes_slide.notes_text_frame:
                text += notes_slide.notes_text_frame.text + "\n"
    return text.strip()

def extract_text_from_ppt(file_path: str) -> str:
    pptx_path = file_path + "x"
    try:
        print("[+] Converting PPT to PPTX...", platform.system())
        if platform.system() == "Windows":
            import comtypes.client
            powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
            powerpoint.Visible = 1
            ppt = powerpoint.Presentations.Open(file_path, WithWindow=False)
            ppt.SaveAs(pptx_path, 24)
            ppt.Close()
            powerpoint.Quit()
        else:
            subprocess.run(["libreoffice", "--headless", "--convert-to", "pptx", file_path, "--outdir", os.path.dirname(file_path)], check=True)
        return extract_text_from_pptx(pptx_path)
    except Exception as e:
        print(f"[!] Error converting or reading PPT: {e}")
        return ""
    finally:
        if os.path.exists(pptx_path):
            os.remove(pptx_path)

def extract_structured_from_excel_or_csv(path):
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, encoding="utf-8", errors='replace')
        return df.to_dict(orient="records")
    elif ext in [".xlsx", ".xls"]:
        if ext == ".xlsx":
            xls = pd.ExcelFile(path, engine="openpyxl")
        else:
            xls = pd.ExcelFile(path, engine="xlrd")
        dfs = {}
        for sheet_name in xls.sheet_names:
            sheet_df = xls.parse(sheet_name)
            for col in sheet_df.columns:
                if pd.api.types.is_datetime64_any_dtype(sheet_df[col]):
                    sheet_df[col] = sheet_df[col].astype(str)
            dfs[sheet_name] = sheet_df.to_dict(orient="records")
        return dfs
    else:
        raise ValueError("Unsupported Excel/CSV file extension!")

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

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_structured_from_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def extract_text_from_xml(file_path: str) -> str:
    tree = ET.parse(file_path)
    root = tree.getroot()
    return ET.tostring(root, encoding='utf-8').decode('utf-8')

def extract_text_from_html(file_path: str) -> str:
    from bs4 import BeautifulSoup
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text()

def extract_text_from_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_yaml(file_path: str) -> str:
    import yaml
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return yaml.dump(data)

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

def insert_document_to_vectorstore(text: str, source_type: str, file_ext: str, document_id: str, user, collection_name):
    qdrant = get_qdrant_vector_store(user, collection_name)
    docs = smart_split_text(text, file_ext)
    print(f"[+] Splitting text into {len(docs)} chunks...")
    global_metadata = extract_metadata(text, file_ext)
    global_metadata["document_id"] = document_id
    for doc in docs:
        if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
            doc.metadata = {}
        doc.metadata.update(global_metadata)
    qdrant.add_documents(docs, batch_size=64)
    print(f"[+] Inserted {len(docs)} chunks into Qdrant collection '{collection_name}'.")

def enrich_document(document_obj, file_text, file_ext):
    try:
        if not file_text.strip():
            return
        summary = summarize_context(file_text[:3000], document_obj.tenant.user)
        metadata = extract_metadata(file_text, file_ext)
        document_obj.summary = summary
        document_obj.keywords = metadata
        document_obj.save()
    except Exception as e:
        print(f"[!] Failed to enrich document: {e}")

def detect_alerts(document_obj, file_text):
    try:
        alert_keywords = [
            "contract expiry", "payment due", "breach of contract", "submission deadline",
            "invoice", "cancellation policy"
        ]
        file_text_lower = file_text.lower()
        for keyword in alert_keywords:
            if keyword in file_text_lower:
                idx = file_text_lower.find(keyword)
                snippet = file_text[max(0, idx-100): idx+100]
                DocumentAlert.objects.create(
                    document=document_obj,
                    keyword=keyword,
                    snippet=snippet
                )
    except Exception as e:
        print(f"[!] Failed to detect alerts: {e}")

def split_text_into_chunks(text: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([text])

def split_csv_rows(text: str, max_lines_per_chunk=50):
    lines = text.strip().split("\n")
    if len(lines) == 0:
        return []
    header = lines[0]
    data_lines = lines[1:]
    chunks = []
    for i in range(0, len(data_lines), max_lines_per_chunk):
        chunk_text = "\n".join([header] + data_lines[i:i + max_lines_per_chunk])
        chunks.append(LangChainDocument(page_content=chunk_text, metadata={"chunk": i // max_lines_per_chunk + 1}))
    return chunks

def split_unstructured_text(text: str, chunk_size=1000, chunk_overlap=200):
    if len(text.strip()) == 0:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents([text])
    for i, doc in enumerate(docs):
        doc.metadata["chunk"] = i + 1
    return docs

def split_json_rows(json_text):
    try:
        data = json.loads(json_text)
        documents = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                documents.append(LangChainDocument(page_content=str(item), metadata={"source": "JSON_item", "chunk": i + 1}))
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        documents.append(LangChainDocument(page_content=str(item), metadata={"source": key, "chunk": i + 1, "type": type(item).__name__}))
                elif isinstance(value, dict):
                    documents.append(LangChainDocument(page_content=str(value), metadata={"source": key, "type": type(value).__name__}))
                elif value is not None:
                    documents.append(LangChainDocument(page_content=str(value), metadata={"source": key, "type": type(value).__name__}))
        return documents
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []

def smart_split_text(text: str, file_ext: str):
    if file_ext == ".json" or (text.startswith('{') and text.endswith('}')):
        return split_json_rows(text)
    elif file_ext in [".csv", ".xls", ".xlsx"]:
        return split_csv_rows(text)
    else:
        return split_unstructured_text(text)

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def summarize_batch(batch_docs: List[LangChainDocument]) -> str:
    combined_context = "\n".join([doc.page_content for doc in batch_docs])
    return summarize_context(combined_context)

def summarize_context(context_text: str, user=None) -> str:
    api_key = get_openai_api_key(user) if user else settings.OPENAI_API_KEY
    llm = ChatOpenAI(api_key=api_key, temperature=0)
    prompt = f"Summarize the following content concisely and accurately:\n\n{context_text}\n\nSummary:"
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)

def batch_documents(documents: List[LangChainDocument], batch_size: int = 5):
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def ask_question(query: str, source_type: str, documents: Optional[List[dict]] = None, user=None) -> str:
    try:
        start = time.time()
        api_key = get_openai_api_key(user) if user else settings.OPENAI_API_KEY
        llm = ChatOpenAI(api_key=api_key, temperature=0)

        system_prompt = """
        You are a helpful document analyst. Your role is to provide accurate, concise, and detailed answers based on the provided document context. 
        Use the information in the documents to answer the question directly and avoid including irrelevant details. 
        If the documents do not contain enough information to answer the question, state so clearly.
        """

        final_context = ""
        max_token_threshold = 4000
        batch_size = 5

        if documents:
            print("[+] Using provided documents as context")
            grouped_docs = {}
            for doc in documents:
                document_id = doc["metadata"].get("document_id", "unknown")
                if document_id not in grouped_docs:
                    grouped_docs[document_id] = []
                grouped_docs[document_id].append(doc)

            mini_summaries = []
            for document_id, docs in grouped_docs.items():
                combined_text = "\n".join([d["content"] for d in docs])
                est_tokens = estimate_tokens(combined_text)
                if est_tokens <= max_token_threshold:
                    mini_summary = summarize_context(combined_text, user)
                else:
                    batches = list(batch_documents(
                        [LangChainDocument(page_content=d["content"]) for d in docs],
                        batch_size=batch_size
                    ))
                    batch_summaries = []
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(summarize_batch, batch) for batch in batches]
                        for future in futures:
                            batch_summaries.append(future.result())
                    combined_summary = "\n".join(batch_summaries)
                    mini_summary = summarize_context(combined_summary, user)
                mini_summaries.append(mini_summary)

            final_context = "\n\n".join(mini_summaries)
        else:
            print("[+] Using default retriever context")
            qdrant = get_qdrant_vector_store(user, user.tenant.collection_name)
            retriever = qdrant.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.get_relevant_documents(query)
            final_context = "\n".join([doc.page_content for doc in retrieved_docs])

        end = time.time()
        print(f"[+] Retrieved and processed context in: {end - start:.2f} seconds")

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

def retrieve_documents_by_vector_id(document_id: str, user, collection_name) -> list:
    try:
        qdrant = get_qdrant_vector_store(user, collection_name)
        filter = Filter(
            must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))]
        )
        search_result = qdrant.client.scroll(
            collection_name=collection_name,
            scroll_filter=filter,
            limit=100,
            with_vectors=False,
            with_payload=True
        )
        if not search_result[0]:
            document = DocumentModel.objects.filter(id=document_id).first()
            if not document:
                return []
            file_ext = Path(document.title).suffix.lower()
            insert_document_to_vectorstore(document.content, "file", file_ext, document_id, user, collection_name)
            search_result = qdrant.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter,
                limit=100,
                with_vectors=False,
                with_payload=True
            )
            if not search_result[0]:
                return []
        documents = [
            {"content": record.payload.get("page_content", ""), "metadata": record.payload.get("metadata", {})}
            for record in search_result[0]
        ]
        return documents
    except Exception as e:
        print(f"[!] Error retrieving documents by document_id: {e}")
        return []

def delete_documents_by_vector_id(document_id: str, user, collection_name) -> bool:
    try:
        qdrant = get_qdrant_vector_store(user, collection_name)
        filter = Filter(must=[FieldCondition(key="metadata.document_id", match=MatchValue(value=document_id))])
        qdrant.client.delete(collection_name=collection_name, points_selector=filter)
        print(f"[+] Successfully deleted Qdrant documents with document_id: {document_id}")
        return True
    except Exception as e:
        print(f"[!] Error deleting documents from Qdrant: {e}")
        return False