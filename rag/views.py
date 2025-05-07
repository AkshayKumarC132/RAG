# rag/views.py

from django.http import JsonResponse
from .models import Tenant,Document, ChatHistory, DocumentAlert, MultiFileChatSession
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
import hashlib
import requests
# logging.basicConfig(filename="app.log",level=logging.INFO)

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
            uploaded_file = serializer.validated_data.get('file', None)
            s3_file_url = serializer.validated_data.get('s3_file_url', None)

            source_type = "file"  # or dynamically decide if needed

            if not uploaded_file and not s3_file_url:
                return Response({"error": "No input file provided. Please upload a local file or provide an S3 file URL."}, status=status.HTTP_400_BAD_REQUEST)

            # Decide which source to use
            if uploaded_file:
                file_name = uploaded_file.name
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
                    for chunk in uploaded_file.chunks():
                        tmp.write(chunk)
                    tmp_path = tmp.name
                print(f"[+] Local file uploaded and saved at: {tmp_path}")

            elif s3_file_url:
                file_name = s3_file_url.split("/")[-1]
                response = requests.get(s3_file_url)
                if response.status_code != 200:
                    return Response({"error": "Failed to download file from S3 URL."}, status=status.HTTP_400_BAD_REQUEST)
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as tmp:
                    print(f"[+] Downloading file from S3 URL: {s3_file_url}")
                    print(f"[+] Saving file as: {file_name}")
                    tmp.write(response.content)
                    print(f"[+] File downloaded successfully.", tmp)
                    tmp_path = tmp.name

                print(f"[+] File downloaded from S3 and saved at: {tmp_path}")
                
            # Generate a unique vector_id
            vector_id = str(uuid.uuid4())


            try:
                # Get tenant from the user's token
                tenant = user.tenant  # Assuming each user has one tenant
                
                extracted_text = extract_text_from_file(tmp_path, file_name)
                file_ext = Path(file_name).suffix.lower()

                if not extracted_text.strip():
                    return Response({"error": "No text could be extracted. Please check if the file is password-protected or empty."}, status=status.HTTP_400_BAD_REQUEST)

                start = time.time()

                insert_document_to_vectorstore(extracted_text, source_type, file_ext, vector_id)

                end = time.time()
                print(f"[+] Embedding and storing in vector store took {end - start:.2f} seconds")

                # Save the document in the Document model
                document = Document(
                    tenant=tenant,
                    title=file_name,
                    content=extracted_text,
                    vector_id=vector_id
                )
                document.save()

                # Enrich document
                self.enrich_document(document, extracted_text, file_ext)
                
                # Detect alerts
                self.detect_alerts(document, extracted_text)
                
                return Response({"message": "File ingested and stored successfully.",
                                 "file name" : file_name,
                                 "document_id": document.id,
                                 "vector_id": vector_id}, status=status.HTTP_200_OK)
            except ValueError as ve:
                return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)
            
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)  # Clean up temp file
                except Exception as e:
                    print(f"[!] Error deleting temp file: {e}")
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        documents = Document.objects.filter(tenant=user.tenant).defer('vector_id').values().order_by('-uploaded_at')
        if not documents:
            return Response({"message": "No documents found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(DocumentSerializer(documents, many=True).data, status=status.HTTP_200_OK)


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

    def detect_alerts(self, document_obj, file_text):
        try:
            alert_keywords = [
                # Contract & Expiry
                "contract expiry", "contract end date", "renewal deadline", "service termination", "expiry notice",
                # Payments
                "payment due", "payment overdue", "invoice overdue", "late fee", "unpaid invoice", "outstanding balance", "collection notice",
                # Legal Risks
                "breach of contract", "penalty clause", "legal action", "non-compliance", "lawsuit", "settlement",
                # Deadlines
                "submission deadline", "due date", "project deadline", "final notice", "critical timeline",
                # Financial
                "advance payment", "refund request", "debit note", "credit note", "balance payable",
                # Risk Specific
                "termination for cause", "default notice", "breach penalty", "financial exposure",
                # Communication
                "no response received", "pending approval", "awaiting confirmation",
                # Supply Chain
                "shipment delay", "logistics issue", "supply disruption",
                # Tax / Regulatory
                "tax penalty", "compliance audit", "regulatory fine",
                # Partner/Vendor Risks
                "partner dispute", "vendor breach", "service level failure",

                # Additional keywords from the provided document
                "invoice", "payment summary", "total amount", "booking fees", "ride charge",
                "cancellation policy", "cancellation fees", "cancellation notice", "cancellation confirmation",
            ]

            file_text_lower = file_text.lower()

            for keyword in alert_keywords:
                if keyword in file_text_lower:
                    idx = file_text_lower.find(keyword)
                    snippet = file_text[max(0, idx-100): idx+100]  # Extract 100 chars before and after
                    DocumentAlert.objects.create(
                        document=document_obj,
                        keyword=keyword,
                        snippet=snippet
                    )
                    print(f"[+] Alert created for '{keyword}' in document {document_obj.title}")

        except Exception as e:
            print(f"[!] Failed to detect alerts: {e}")


class AskAPIView(generics.CreateAPIView):
    serializer_class = AskQuestionSerializer
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        # user = auth_token.user
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            vector_id = serializer.validated_data.get('vector_id')  # Optional field in serializer
            chat_history = serializer.validated_data.get('chat_history')  # Optional field in serializer
            user_identifier = serializer.validated_data['user_identifier']
            source_type = "file"  # or dynamic based on your design

            if not question:
                return Response(
                    {"error": "A question must be provided."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            # Validate document access
            if vector_id:
                document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
                if user_identifier != user.email:  # Admin uses email as identifier
                    if not DocumentAccess.objects.filter(document=document, user_identifier=user_identifier).exists():
                        return Response({'error': 'No access to this document.'}, status=status.HTTP_403_FORBIDDEN)
            try:
                if vector_id:
                    # Retrieve documents by vector_id
                    documents = retrieve_documents_by_vector_id(vector_id)
                    print(f"[+] Retrieved documents for vector_id {vector_id}: {documents}")
                    if not documents:
                        return Response({"error": "No documents found for the given vector_id."}, status=status.HTTP_404_NOT_FOUND)
                    answer = ask_question_for_single_document(question, source_type=source_type, documents=documents, chat_history=chat_history)
                    # return Response({"vector_id": vector_id, "documents": documents}, status=status.HTTP_200_OK)
                    # Update chat history
                    chat, _ = ChatHistory.objects.get_or_create(
                        vector_id=vector_id,
                        user_identifier=user_identifier,
                        tenant=user.tenant,
                        defaults={'history': []}
                    )
                    chat.history.append({"role": "user", "content": question})
                    chat.history.append({"role": "assistant", "content": answer})
                    chat.save()
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

        document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
        documents = retrieve_documents_by_vector_id(vector_id)
        if not documents:
            return Response({"error": "No documents found."}, status=status.HTTP_404_NOT_FOUND)
        return Response({"vector_id": vector_id,"documents": documents}, status=status.HTTP_200_OK)


class DeleteDocumentAPIView(APIView):
    def delete(self, request, token, vector_id):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)

        try:
            delete_documents_by_vector_id(vector_id)
            document.delete()
            return Response({"message": "Document deleted successfully."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        

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
    

class MultiFileAskAPIView(generics.CreateAPIView):
    serializer_class = AskQuestionSerializer

    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            vector_ids = request.data.get('vector_ids', [])
            chat_history = serializer.validated_data.get('chat_history')
            user_identifier = serializer.validated_data['user_identifier']

            if not question:
                return Response({"error": "A question must be provided."}, status=status.HTTP_400_BAD_REQUEST)
            if not vector_ids:
                return Response({"error": "vector_ids are required."}, status=status.HTTP_400_BAD_REQUEST)

            # Validate document access
            for vector_id in vector_ids:
                document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
                if user_identifier != user.email:  # Admin uses email as identifier
                    if not DocumentAccess.objects.filter(document=document, user_identifier=user_identifier).exists():
                        return Response({'error': f'No access to document {vector_id}.'}, status=status.HTTP_403_FORBIDDEN)

            try:
                documents = retrieve_documents_by_vector_ids(vector_ids)
                if not documents:
                    return Response({"error": "No documents found for the given vector_ids."}, status=status.HTTP_404_NOT_FOUND)

                answer = ask_question(
                    question,
                    source_type="file",
                    documents=documents,
                    chat_history=chat_history
                )

                # Update or create session
                sorted_vector_ids = sorted(vector_ids)
                vector_hash = hashlib.sha256(json.dumps(sorted_vector_ids).encode('utf-8')).hexdigest()
                session, created = MultiFileChatSession.objects.get_or_create(
                    vector_hash=vector_hash,
                    user_identifier=user_identifier,
                    tenant=user.tenant,
                    defaults={
                        'session_id': str(uuid.uuid4()),
                        'user': user,
                        'vector_ids': vector_ids,
                        'history': [{'question': question, 'answer': answer}]
                    }
                )

                if not created:
                    session.history.append({'question': question, 'answer': answer})
                    session.save()

                return Response({
                    'session_id': session.session_id,
                    'answer': answer
                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GlobalAskAPIView(generics.CreateAPIView):
    serializer_class = AskQuestionSerializer

    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            chat_history = serializer.validated_data.get('chat_history')
            user_identifier = serializer.validated_data['user_identifier']

            if not question:
                return Response({"error": "A question must be provided."}, status=status.HTTP_400_BAD_REQUEST)

            try:
                documents = Document.objects.filter(tenant=user.tenant)
                vector_ids = [doc.vector_id for doc in documents]
                if not vector_ids:
                    return Response({"error": "No documents found for this tenant."}, status=status.HTTP_404_NOT_FOUND)

                retrieved_docs = retrieve_documents_by_vector_ids(vector_ids)
                answer = ask_question(
                    question,
                    source_type="file",
                    documents=retrieved_docs,
                    chat_history=chat_history
                )

                return Response({"answer": answer}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def get_document_alerts(request, vector_id):
    document = get_object_or_404(Document, vector_id=vector_id)
    alerts = DocumentAlert.objects.filter(document=document)
    alert_data = [{"keyword": alert.keyword, "snippet": alert.snippet, "created_at": alert.created_at} for alert in alerts]
    return Response({"alerts": alert_data}, status=status.HTTP_200_OK)

def compute_vector_hash(vector_ids: list[str]) -> str:
    sorted_ids = sorted(vector_ids)
    return hashlib.sha256(",".join(sorted_ids).encode()).hexdigest()


# This endpoint is for saving and retrieving chat history for multiple files
@api_view(['GET', 'POST', 'DELETE'])
def chat_history(request, token, vector_id):
    auth_token = get_object_or_404(AuthToken, token_key=token)
    user = auth_token.user
    user_identifier = request.query_params.get('user_identifier') or request.data.get('user_identifier')

    if not user_identifier:
        return Response({'error': 'user_identifier is required.'}, status=status.HTTP_400_BAD_REQUEST)

    # Validate document access
    document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
    if user_identifier != user.email:  # Admin uses email as identifier
        if not DocumentAccess.objects.filter(document=document, user_identifier=user_identifier).exists():
            return Response({'error': 'No access to this document.'}, status=status.HTTP_403_FORBIDDEN)

    try:
        chat = ChatHistory.objects.get(vector_id=vector_id, user_identifier=user_identifier, tenant=user.tenant)
    except ChatHistory.DoesNotExist:
        chat = None

    if request.method == 'GET':
        if chat:
            return Response({'history': chat.history}, status=status.HTTP_200_OK)
        return Response({'history': []}, status=status.HTTP_200_OK)

    elif request.method == 'POST':
        history = request.data.get('history', [])
        if chat:
            chat.history = history
            chat.save()
        else:
            chat = ChatHistory.objects.create(
                vector_id=vector_id,
                user_identifier=user_identifier,
                history=history,
                tenant=user.tenant
            )
        return Response({'message': 'Chat history saved.'}, status=status.HTTP_200_OK)

    elif request.method == 'DELETE':
        if chat:
            chat.delete()
            return Response({'message': 'Chat history cleared.'}, status=status.HTTP_200_OK)
        return Response({'message': 'No chat history found.'}, status=status.HTTP_200_OK)


@api_view(['GET', 'POST', 'DELETE'])
def chat_history_multifile(request, token):
    auth_token = get_object_or_404(AuthToken, token_key=token)
    user = auth_token.user
    user_identifier = request.query_params.get('user_identifier') or request.data.get('user_identifier')

    if not user_identifier:
        return Response({'error': 'user_identifier is required.'}, status=status.HTTP_400_BAD_REQUEST)

    if request.method == 'POST':
        vector_ids = request.data.get('vector_ids', [])
        history = request.data.get('history', [])

        if not vector_ids:
            return Response({'error': 'vector_ids are required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Validate document access
        for vector_id in vector_ids:
            document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)
            if user_identifier != user.email:  # Admin uses email as identifier
                if not DocumentAccess.objects.filter(document=document, user_identifier=user_identifier).exists():
                    return Response({'error': f'No access to document {vector_id}.'}, status=status.HTTP_403_FORBIDDEN)

        # Generate vector_hash
        sorted_vector_ids = sorted(vector_ids)
        vector_hash = hashlib.sha256(json.dumps(sorted_vector_ids).encode('utf-8')).hexdigest()

        # Create or update session
        session, created = MultiFileChatSession.objects.get_or_create(
            vector_hash=vector_hash,
            user_identifier=user_identifier,
            tenant=user.tenant,
            defaults={
                'session_id': str(uuid.uuid4()),
                'user': user,
                'vector_ids': vector_ids,
                'history': history
            }
        )

        if not created:
            session.vector_ids = vector_ids
            session.history = history
            session.save()

        return Response({'session_id': session.session_id, 'message': 'Session saved.'}, status=status.HTTP_200_OK)

    elif request.method == 'GET':
        session_id = request.query_params.get('session_id')
        if not session_id:
            return Response({'error': 'session_id is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session = MultiFileChatSession.objects.get(
                session_id=session_id,
                user_identifier=user_identifier,
                tenant=user.tenant
            )
            return Response(MultiFileChatSessionSerializer(session).data, status=status.HTTP_200_OK)
        except MultiFileChatSession.DoesNotExist:
            return Response({'error': 'Session not found.'}, status=status.HTTP_404_NOT_FOUND)

    elif request.method == 'DELETE':
        session_id = request.data.get('session_id')
        if not session_id:
            return Response({'error': 'session_id is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            session = MultiFileChatSession.objects.get(
                session_id=session_id,
                user_identifier=user_identifier,
                tenant=user.tenant
            )
            session.delete()
            return Response({'message': 'Session deleted.'}, status=status.HTTP_200_OK)
        except MultiFileChatSession.DoesNotExist:
            return Response({'error': 'Session not found.'}, status=status.HTTP_404_NOT_FOUND)

class ShareDocumentAPIView(APIView):
    # permission_classes = [IsAdmin]  # Uncomment when RBAC is implemented
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        print(user)

        vector_id = request.data.get('vector_id')
        user_identifier = request.data.get('user_identifier')  # e.g., TM email

        if not vector_id or not user_identifier:
            return Response({"error": "vector_id and user_identifier are required."}, status=status.HTTP_400_BAD_REQUEST)

        document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)

        access, created = DocumentAccess.objects.get_or_create(
            document=document,
            user_identifier=user_identifier,
            defaults={'granted_by': user}
        )

        if not created:
            return Response({"message": "Access already granted."}, status=status.HTTP_200_OK)

        return Response({"message": f"Access granted to {user_identifier} for document {vector_id}."}, status=status.HTTP_201_CREATED)
    
    def get(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user

        user_identifier = request.data.get('user_identifier')  # e.g., TM email

        if not user_identifier:
            return Response({"error": "user_identifier is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Query shared documents
        access_records = DocumentAccess.objects.filter(
            user_identifier=user_identifier,
            document__tenant=user.tenant
        )
        documents = [access.document for access in access_records]
        
        if not documents:
            return Response({"message": "No shared documents found.", "shared_documents": []}, status=status.HTTP_200_OK)

        serializer = DocumentSerializer(documents, many=True)
        return Response({
            "user_identifier": user_identifier,
            "shared_documents": serializer.data
            }, status=status.HTTP_200_OK)


class RemoveDocumentAccessAPIView(generics.DestroyAPIView):

    def put(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        vector_id = request.data.get('vector_id')
        user_identifier = request.data.get('user_identifier')

        if not vector_id or not user_identifier:
            return Response({"error": "vector_id and user_identifier are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Validate document access
        document = get_object_or_404(Document, vector_id=vector_id, tenant=user.tenant)

        try:
            access = DocumentAccess.objects.get(document=document, user_identifier=user_identifier)
            access.delete()
            # Update MultiFileChatSession to remove vector_id
            self.update_multi_file_session(user_identifier, vector_id, user.tenant)
            return Response({"message": "Access removed successfully for document {vector_id}."}, status=status.HTTP_200_OK)
        except DocumentAccess.DoesNotExist:
            return Response({"error": "Access does not exist."}, status=status.HTTP_400_BAD_REQUEST)
        
    def update_multi_file_session(self,user_identifier, vector_id, tenant):
        """Remove vector_id from MultiFileChatSession and update vector_hash."""
        sessions = MultiFileChatSession.objects.filter(user_identifier=user_identifier, tenant=tenant)
        for session in sessions:
            if vector_id in session.vector_ids:
                session.vector_ids.remove(vector_id)
                if session.vector_ids:  # Update hash if vector_ids is not empty
                    session.vector_hash = hashlib.sha256(
                        ''.join(sorted(session.vector_ids)).encode('utf-8')
                    ).hexdigest()
                else:  # Clear hash if no vector_ids remain
                    session.vector_hash = None
                session.save()