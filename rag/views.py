# rag/views.py

from django.http import JsonResponse
from .models import Tenant,Document, ChatHistory
import json
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from knox.models import AuthToken
from rest_framework import generics
from .serializers import *
from django.contrib.auth import authenticate, get_user_model
from django.shortcuts import get_object_or_404
from rest_framework.parsers import MultiPartParser, FormParser
from .utils import extract_text_from_file, insert_document_to_vectorstore, ask_question, retrieve_documents_by_vector_id,delete_documents_by_vector_id, retrieve_documents_by_vector_ids,summarize_context, extract_metadata,ask_question_for_single_document
import tempfile
from pathlib import Path
import time
import logging
import uuid

logging.basicConfig(filename="app.log",level=logging.INFO)

User = get_user_model()

class ProtectedView(APIView):

    def get(self, request,token):
        if not token:
            return Response({'error': 'Token not provided.'}, status=status.HTTP_400_BAD_REQUEST)
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        return Response({
            'user': UserSerializer(user).data,
            'token': token,
            'expiry': auth_token.expiry
        }, status=status.HTTP_200_OK)

class RegisterAPI(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer

class LoginView(generics.CreateAPIView):
    serializer_class = LoginSerializer

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        username = data.get('username')
        password = data.get('password')

        if username is None or password is None:
            return Response({'error': 'Please provide both username and password.'},
                            status=status.HTTP_400_BAD_REQUEST)

        user = authenticate(request, username=username, password=password)

        if not user:
            return Response({'error': 'Invalid credentials.'},
                            status=status.HTTP_401_UNAUTHORIZED)

        # Generate token
        token_instance, token = AuthToken.objects.create(user)

        # Serialize user data
        user_data = UserSerializer(user).data

        return Response({
            'token': token_instance.token_key,
            'expiry': token_instance.expiry,
            'user': user_data
        }, status=status.HTTP_200_OK)

class LogoutView(APIView):
    def post(self, request, token=None, format=None):
        if not token:
            return Response({'error': 'Token not provided.'}, status=status.HTTP_400_BAD_REQUEST)

        auth_token = get_object_or_404(AuthToken, token_key=token)
        auth_token.delete()
        return Response({'message': 'Logged out successfully.'}, status=status.HTTP_204_NO_CONTENT)
    

class IngestAPIView(generics.CreateAPIView):
    serializer_class = IngestDocumentSerializer
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            source_type = "file"  # or dynamically decide if needed

            # Generate a unique vector_id
            vector_id = str(uuid.uuid4())

            
            # Save to a temp file
            # with tempfile.NamedTemporaryFile(delete=False) as tmp:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            print(f"[+] Temporary file created at: {tmp_path}")
            try:
                # Get tenant from the user's token
                tenant = user.tenant  # Assuming each user has one tenant
                
                extracted_text = extract_text_from_file(tmp_path, uploaded_file.name)
                file_ext = Path(uploaded_file.name).suffix.lower()

                start = time.time()

                insert_document_to_vectorstore(extracted_text, source_type, file_ext, vector_id)

                end = time.time()
                print(f"[+] Embedding and storing in vector store took {end - start:.2f} seconds")

                # Save the document in the Document model
                document = Document(
                    tenant=tenant,
                    title=uploaded_file.name,
                    content=extracted_text,
                    vector_id=vector_id
                )
                document.save()

                self.enrich_document(document, extracted_text, file_ext)

                return Response({"message": "File ingested and stored successfully.",
                                 "file name" : uploaded_file.name,
                                 "document_id": document.id,
                                 "vector_id": vector_id}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            # finally:
            #     Path(tmp_path).unlink(missing_ok=True)  # Clean up temp file
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        documents = Document.objects.filter(tenant=user.tenant).defer('vector_id').values()
        if not documents:
            return Response({"message": "No documents found."}, status=status.HTTP_404_NOT_FOUND)
        return Response({"documents": list(documents)}, status=status.HTTP_200_OK)


    def enrich_document(self,document_obj, file_text, file_ext):
        try:
            if not file_text.strip():
                print("[!] Empty file content, skipping enrichment")
                return

            # Summarize the document (first 3000 chars)
            summary = summarize_context(file_text[:3000])
            # Extract keywords/metadata
            metadata = extract_metadata(file_text, file_ext)

            document_obj.summary = summary
            document_obj.keywords = metadata
            document_obj.save()

            print(f"[+] Enriched document {document_obj.title} with summary and keywords.")

        except Exception as e:
            print(f"[!] Failed to enrich document: {e}")

class AskAPIView(generics.CreateAPIView):
    serializer_class = AskQuestionSerializer
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        # user = auth_token.user
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            vector_id = serializer.validated_data.get('vector_id')  # Optional field in serializer
            chat_history = serializer.validated_data.get('chat_history')  # Optional field in serializer
            source_type = "file"  # or dynamic based on your design

            if not question:
                return Response(
                    {"error": "A question must be provided."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                if vector_id:
                    # Retrieve documents by vector_id
                    documents = retrieve_documents_by_vector_id(vector_id)
                    print(f"[+] Retrieved documents for vector_id {vector_id}: {documents}")
                    if not documents:
                        return Response({"error": "No documents found for the given vector_id."}, status=status.HTTP_404_NOT_FOUND)
                    answer = ask_question_for_single_document(question, source_type=source_type, documents=documents, chat_history=chat_history)
                    # return Response({"vector_id": vector_id, "documents": documents}, status=status.HTTP_200_OK)
                else:
                    answer = ask_question(question, source_type,chat_history)
                return Response({"answer": answer}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        

class RetrieveByVectorIdAPIView(APIView):
    def get(self, request, token, vector_id):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user

        try:
            doc = Document.objects.get(vector_id=vector_id, tenant=user.tenant)
            if not doc:
                return Response({"error": "Document not found."}, status=status.HTTP_404_NOT_FOUND)
            # Retrieve documents from Qdrant by vector_id
            documents = retrieve_documents_by_vector_id(vector_id)

            if not documents:
                return Response({"error": "No documents found for the given vector_id."}, status=status.HTTP_404_NOT_FOUND)

            return Response({
                "vector_id": vector_id,
                "documents": documents
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DeleteDocumentAPIView(APIView):
    def delete(self, request, token, vector_id):
        # Validate authentication token
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user

        try:
            # Find the document in the database
            document = Document.objects.filter(vector_id=vector_id, tenant=user.tenant).first()
            if not document:
                print(f"[!] No document found with vector_id: {vector_id} for tenant: {user.tenant.id}")
                return Response(
                    {"error": "No document found with the given vector_id."},
                    status=status.HTTP_404_NOT_FOUND
                )

            is_deleted = delete_documents_by_vector_id(vector_id)
            print("[+] Deleting document from vector store...", is_deleted)
            if not is_deleted:
                print(f"[!] Failed to delete document with vector_id: {vector_id} from vector store")
                return Response(
                    {"error": "Failed to delete document from vector store."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            # Delete from Django database
            document_title = document.title
            document.delete()
            print(f"[+] Successfully deleted document '{document_title}' from database")

            return Response(
                {
                    "message": f"Document '{document_title}' with vector_id {vector_id} deleted successfully.",
                    "vector_id": vector_id
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            print(f"[!] Error deleting document with vector_id {vector_id}: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        

@api_view(['GET', 'POST', 'DELETE'])
def chat_history(request, token,vector_id):
    auth_token = get_object_or_404(AuthToken, token_key=token)
    user = auth_token.user
    if not vector_id:
        return Response({'error': 'Vector ID not provided.'}, status=status.HTTP_400_BAD_REQUEST)
    
    if not user:
        return Response({'error': 'User not authenticated.'}, status=status.HTTP_401_UNAUTHORIZED)

    try:
        chat = ChatHistory.objects.get(vector_id=vector_id)
    except ChatHistory.DoesNotExist:
        chat = None

    if request.method == 'GET':
        if chat:
            return Response({'history': chat.history})
        else:
            return Response({'history': []})

    elif request.method == 'POST':
        history = request.data.get('history', [])
        if chat:
            chat.history = history
            chat.save()
        else:
            ChatHistory.objects.create(vector_id=vector_id, history=history)
        return Response({'message': 'Chat history saved.'})

    elif request.method == 'DELETE':
        if chat:
            chat.delete()
        return Response({'message': 'Chat history cleared.'})
    

class MultiFileAskAPIView(APIView):
    """
    Accepts a question and list of vector_ids to perform multi-file RAG
    """
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        data = request.data

        question = data.get("question")
        vector_ids = data.get("vector_ids", [])

        if not question:
            return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not vector_ids:
            return Response({"error": "At least one vector_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            documents = retrieve_documents_by_vector_ids(vector_ids)
            if not documents:
                return Response({"error": "No documents found for the provided vector_ids."}, status=404)

            answer = ask_question(question, source_type="file", documents=documents)
            return Response({"answer": answer}, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)


class GlobalAskAPIView(APIView):
    """
    Accepts a question and answers it based on *all documents* uploaded by the current user
    """
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        data = request.data

        question = data.get("question")
        if not question:
            return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Fetch all vector_ids for the user's tenant
            vector_ids = list(Document.objects.filter(tenant=user.tenant).values_list("vector_id", flat=True))
            if not vector_ids:
                return Response({"error": "No documents found for this tenant."}, status=404)

            documents = retrieve_documents_by_vector_ids(vector_ids)
            if not documents:
                return Response({"error": "No documents found in Qdrant."}, status=404)

            answer = ask_question(question, source_type="file", documents=documents)
            return Response({"answer": answer}, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)


