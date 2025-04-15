import io
import docx
import PyPDF2
from django.core.files.uploadedfile import InMemoryUploadedFile
from rest_framework.exceptions import ValidationError
import textract

def process_file(file):
    """
    Extract text from various types of documents.
    Supports PDFs, DOCX, and others. Add more file types as needed.
    """

    # Get the file extension
    ext = file.name.split('.')[-1].lower()
    print("Extension: ", ext)

    if ext == 'pdf':
        return extract_text_from_pdf(file)
    elif ext == 'docx':
        return extract_text_from_docx(file)
    else:
        try:
            # If the file is an InMemoryUploadedFile, read its content in memory
            if isinstance(file, InMemoryUploadedFile):
                # Create a file-like object from the uploaded file
                file_stream = io.BytesIO(file.read())
                print("File stream: ", file_stream)
                print("File name: ", file.name)
                print("File size: ", file.size)
                print("File type: ", file.content_type)
                print()
                text = textract.process(file_stream)  # Process the BytesIO object
                return text.decode('utf-8')  # Decode the result to a string
            else:
                # For other types of files, use the temporary file path
                text = textract.process(file.temporary_file_path()).decode('utf-8')
                return text
        except Exception as e:
            raise ValidationError(f"Error processing file: {str(e)}")

def extract_text_from_pdf(pdf_file):
    try:
        # PDF extraction
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise ValidationError(f"Error processing PDF file: {str(e)}")

def extract_text_from_docx(docx_file):
    try:
        # DOCX extraction
        doc = docx.Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        raise ValidationError(f"Error processing DOCX file: {str(e)}")