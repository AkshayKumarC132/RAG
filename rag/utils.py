import os
import docx
import PyPDF2
from django.core.files.uploadedfile import InMemoryUploadedFile
from rest_framework.exceptions import ValidationError
import textract
import tempfile
import pytesseract
from PIL import Image
import io

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
    elif ext in ['png', 'jpg', 'jpeg']:
        try:
            # Convert InMemoryUploadedFile to image
            img = Image.open(file)
            
            # Optional: Preprocess image for better OCR
            img = img.convert('L')  # Convert to grayscale
            # Enhance contrast or resize if needed (uncomment if required)
            # img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            raise ValidationError(f"Error processing image: {str(e)}")
    else:
        try:
            # Handle InMemoryUploadedFile or other uploaded files
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_file:
                # Write the file content to the temporary file
                for chunk in file.chunks():
                    temp_file.write(chunk)
                temp_file.flush()
                temp_file_path = temp_file.name

            # Process the file using textract
            text = textract.process(temp_file_path)
            text = text.decode('utf-8')

            # Clean up the temporary file
            os.unlink(temp_file_path)

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