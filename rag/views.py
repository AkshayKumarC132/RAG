from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError
from knox.models import AuthToken
from django.shortcuts import get_object_or_404
from .models import Tenant, Document, VectorStore, Assistant, Thread, Message, Run, DocumentAlert, DocumentAccess, OpenAIKey
from .serializers import *
from .utils import (
    process_file, extract_text_from_file, insert_document_to_vectorstore,
    ask_question, retrieve_documents_by_vector_id, delete_documents_by_vector_id,
    enrich_document, detect_alerts, get_authenticated_user
)
import threading
from datetime import datetime
from django.contrib.auth import get_user_model
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
User = get_user_model()

class TokenAuthenticatedMixin:
    def initial(self, request, *args, **kwargs):
        try:
            authenticated_user = get_authenticated_user(kwargs.get('token'))
            # Update request.user to ensure serializer context has the authenticated user
            request.user = authenticated_user
            self.user = authenticated_user
            super().initial(request, *args, **kwargs)
        except ValueError as e:
            logger.error(f"Authentication failed: {e}")
            return Response({"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED)

# Tenant CRUD
class TenantCreateAPIView(generics.CreateAPIView):
    serializer_class = TenantSerializer
    queryset = Tenant.objects.all()

class TenantListAPIView(generics.ListAPIView):
    serializer_class = TenantSerializer
    queryset = Tenant.objects.all()

class TenantRetrieveUpdateDestroyAPIView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = TenantSerializer
    queryset = Tenant.objects.all()
    lookup_field = 'id'

class RegisterAPI(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer

class UserListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = UserSerializer
    def get_queryset(self):
        return User.objects.filter(tenant=self.user.tenant)

class UserRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = UserSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return User.objects.filter(tenant=self.user.tenant)

# Authentication
class LoginView(generics.CreateAPIView):
    serializer_class = LoginSerializer

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        if not hasattr(user, 'tenant') or user.tenant is None:
            logger.error(f"User {user.username} has no associated tenant")
            return Response({"error": "User has no associated tenant"}, status=status.HTTP_403_FORBIDDEN)
        token_instance, token = AuthToken.objects.create(user)
        return Response({
            'token': token_instance.token_key,
            'user': UserSerializer(user).data
        }, status=status.HTTP_201_CREATED)

class LogoutView(TokenAuthenticatedMixin, APIView):
    def post(self, request, token=None, format=None):
        try:
            auth_token = get_object_or_404(AuthToken, token_key=token)
            auth_token.delete()
            return Response({'message': 'Logged out successfully.'}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error logging out: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class ProtectedView(TokenAuthenticatedMixin, APIView):
    def get(self, request, token):
        return Response({
            'user': UserSerializer(self.user).data,
            'token': token,
        }, status=status.HTTP_200_OK)

# OpenAIKey CRUD
class OpenAIKeyCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = OpenAIKeySerializer
    def perform_create(self, serializer):
        serializer.save(user=self.user)

class OpenAIKeyListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = OpenAIKeySerializer
    def get_queryset(self):
        return OpenAIKey.objects.filter(user=self.user)

class OpenAIKeyRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = OpenAIKeySerializer
    lookup_field = 'id'
    def get_queryset(self):
        return OpenAIKey.objects.filter(user=self.user)

# VectorStore CRUD
class VectorStoreCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = VectorStoreSerializer
    def perform_create(self, serializer):
        serializer.save(tenant=self.user.tenant)

class VectorStoreListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = VectorStoreSerializer
    def get_queryset(self):
        return VectorStore.objects.filter(tenant=self.user.tenant)

class VectorStoreRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = VectorStoreSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return VectorStore.objects.filter(tenant=self.user.tenant)

# Document CRUD
class IngestAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = IngestDocumentSerializer

    def post(self, request, token):
        try:
            serializer = self.serializer_class(data=request.data)
            serializer.is_valid(raise_exception=True)
            uploaded_file = serializer.validated_data.get('file')
            s3_file_url = serializer.validated_data.get('s3_file_url')
            vector_store_id = serializer.validated_data['vector_store_id']

            # Validate tenant
            if not hasattr(self.user, 'tenant') or self.user.tenant is None:
                logger.error(f"User {self.user.username} has no associated tenant")
                return Response({"error": "User has no associated tenant"}, status=status.HTTP_403_FORBIDDEN)

            vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=self.user.tenant)

            tmp_path, file_name = process_file(uploaded_file, s3_file_url)
            extracted_text = extract_text_from_file(tmp_path, file_name)
            file_ext = Path(file_name).suffix.lower()

            if not extracted_text.strip():
                logger.warning(f"No text extracted from file {file_name}")
                return Response({"error": "No text could be extracted from the file."}, status=status.HTTP_400_BAD_REQUEST)

            document = Document(
                tenant=self.user.tenant,
                vector_store=vector_store,
                title=file_name,
                content=extracted_text
            )
            document.save()

            insert_document_to_vectorstore(
                extracted_text,
                "file",
                file_ext,
                str(document.id),
                user=self.user,
                collection_name=self.user.tenant.collection_name,
                vector_store_id=str(vector_store.id)
            )

            enrich_document(document, extracted_text, file_ext)
            detect_alerts(document, extracted_text)

            logger.info(f"Successfully ingested document {document.id}: {file_name}")
            return Response({
                "message": "File ingested successfully.",
                "file_name": file_name,
                "document_id": document.id,
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        finally:
            if 'tmp_path' in locals():
                Path(tmp_path).unlink(missing_ok=True)

class DocumentListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = DocumentSerializer
    def get_queryset(self):
        vector_store_id = self.request.query_params.get('vector_store_id')
        queryset = Document.objects.filter(tenant=self.user.tenant)
        if vector_store_id:
            vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=self.user.tenant)
            queryset = queryset.filter(vector_store=vector_store)
        return queryset

class DocumentRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = DocumentSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Document.objects.filter(tenant=self.user.tenant)

    def perform_destroy(self, instance):
        delete_documents_by_vector_id(str(instance.id), self.user, instance.tenant.collection_name)
        instance.delete()

# Assistant CRUD
class AssistantCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = AssistantSerializer
    def perform_create(self, serializer):
        vector_store_id = serializer.validated_data.get('vector_store_id')
        vector_store = None
        if vector_store_id:
            vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=self.user.tenant)
        serializer.save(tenant=self.user.tenant, vector_store=vector_store, creator=self.user)

class AssistantListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = AssistantSerializer
    def get_queryset(self):
        return Assistant.objects.filter(tenant=self.user.tenant)

class AssistantRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = AssistantSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Assistant.objects.filter(tenant=self.user.tenant)

class ThreadCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = ThreadSerializer
    def perform_create(self, serializer):
        vector_store_id = serializer.validated_data['vector_store_id']
        vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=self.user.tenant)
        serializer.save(tenant=self.user.tenant, vector_store=vector_store)

class ThreadListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = ThreadSerializer
    def get_queryset(self):
        return Thread.objects.filter(tenant=self.user.tenant)

class ThreadRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = ThreadSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Thread.objects.filter(tenant=self.user.tenant)

class ThreadMessagesAPIView(TokenAuthenticatedMixin, APIView):
    def get(self, request, token, thread_id):
        try:
            thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant)
            messages = thread.messages.order_by('created_at')
            return Response(MessageSerializer(messages, many=True).data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error retrieving thread messages: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class MessageCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = MessageSerializer
    def perform_create(self, serializer):
        thread_id = serializer.validated_data['thread_id']
        thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant)
        serializer.save(thread=thread, role = 'user')

class MessageListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = MessageSerializer
    def get_queryset(self):
        thread_id = self.request.query_params.get('thread_id')
        queryset = Message.objects.filter(thread__tenant=self.user.tenant)
        if thread_id:
            thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant)
            queryset = queryset.filter(thread=thread)
        return queryset

class MessageRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = MessageSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Message.objects.filter(thread__tenant=self.user.tenant)

# class RunCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
#     serializer_class = RunSerializer

#     def perform_create(self, serializer):
#         try:
#             thread_id = serializer.validated_data['thread_id']
#             assistant_id = serializer.validated_data['assistant_id']
#             thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant)
#             assistant = get_object_or_404(Assistant, id=assistant_id, tenant=self.user.tenant)

#             # Check for user messages
#             messages = thread.messages.order_by('created_at')
#             user_messages = [msg.content for msg in messages if msg.role == 'user']
#             if not user_messages:
#                 raise ValidationError({"error": "No user messages found in the thread."})

#             # Determine vector store
#             vector_store = assistant.vector_store
#             if not vector_store:
#                 vector_store = thread.vector_store
#                 if not vector_store:
#                     vector_store = VectorStore.objects.filter(tenant=self.user.tenant).first()
#                     if not vector_store:
#                         raise ValidationError({"error": "No vector store available."})

#             # Check for documents
#             documents = retrieve_documents_by_vector_id(
#                 str(vector_store.id),
#                 user=self.user,
#                 collection_name=self.user.tenant.collection_name
#             )
#             if not documents:
#                 raise ValidationError({
#                     "error": f"No documents are assigned to the vector store {vector_store.id}."
#                 })

#             run = serializer.save(thread=thread, assistant=assistant, status='queued')
#             threading.Thread(target=self.process_run, args=(run,)).start()
#             logger.info(f"Started background processing for run {run.id}")
#         except ValidationError as e:
#             logger.error(f"Run creation failed: {e}")
#             raise
#         except Exception as e:
#             logger.error(f"Error creating run: {e}")
#             raise ValidationError({"error": f"Failed to create run: {str(e)}"})

#     def process_run(self, run):
#         try:
#             run.status = 'in_progress'
#             run.save()
#             thread = run.thread
#             assistant = run.assistant
#             messages = thread.messages.order_by('created_at')
#             user_messages = [msg.content for msg in messages if msg.role == 'user']
#             query = user_messages[-1]
#             vector_store = assistant.vector_store or thread.vector_store or VectorStore.objects.filter(tenant=self.user.tenant).first()
#             documents = retrieve_documents_by_vector_id(
#                 str(vector_store.id),
#                 user=self.user,
#                 collection_name=self.user.tenant.collection_name
#             )
#             answer = ask_question(
#                 query,
#                 "file",
#                 documents=documents,
#                 user=self.user,
#                 collection_name=self.user.tenant.collection_name
#             )
#             Message.objects.create(thread=thread, role='assistant', content=answer)
#             run.status = 'completed'
#             run.completed_at = datetime.now()
#             run.save()
#             logger.info(f"Run {run.id} completed successfully")
#         except Exception as e:
#             run.status = 'failed'
#             run.save()
#             Message.objects.create(
#                 thread=thread,
#                 role='assistant',
#                 content=f'Run failed due to an error: {str(e)}'
#             )
#             logger.error(f"Run {run.id} processing failed: {e}")

class RunCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = RunSerializer

    def perform_create(self, serializer):
        try:
            thread_id = serializer.validated_data['thread_id']
            assistant_id = serializer.validated_data['assistant_id']
            thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant)
            assistant = get_object_or_404(Assistant, id=assistant_id, tenant=self.user.tenant)

            # Check for user messages
            messages = thread.messages.order_by('created_at')
            user_messages = [msg.content for msg in messages if msg.role == 'user']
            if not user_messages:
                raise ValidationError({"error": "No user messages found in the thread."})

            # Determine vector store
            vector_store = assistant.vector_store
            if not vector_store:
                vector_store = thread.vector_store
                if not vector_store:
                    vector_store = VectorStore.objects.filter(tenant=self.user.tenant).first()
                    if not vector_store:
                        raise ValidationError({"error": "No vector store available."})

            # Check for documents
            documents = retrieve_documents_by_vector_id(
                str(vector_store.id),
                user=self.user,
                collection_name=self.user.tenant.collection_name
            )
            if not documents:
                raise ValidationError({
                    "error": f"No documents are assigned to the vector store {vector_store.id}."
                })

            run = serializer.save(thread=thread, assistant=assistant, status='queued')
            threading.Thread(target=self.process_run, args=(run,)).start()
            logger.info(f"Started background processing for run {run.id}")
        except ValidationError as e:
            logger.error(f"Run creation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating run: {e}")
            raise ValidationError({"error": f"Failed to create run: {str(e)}"})

    def process_run(self, run):
        try:
            run.status = 'in_progress'
            run.save()
            thread = run.thread
            assistant = run.assistant
            messages = thread.messages.order_by('created_at')
            user_messages = [msg.content for msg in messages if msg.role == 'user']
            query = user_messages[-1]
            vector_store = assistant.vector_store or thread.vector_store or VectorStore.objects.filter(tenant=self.user.tenant).first()
            documents = retrieve_documents_by_vector_id(
                str(vector_store.id),
                user=self.user,
                collection_name=self.user.tenant.collection_name
            )
            answer = ask_question(
                query,
                "file",
                documents=documents,
                user=self.user,
                collection_name=self.user.tenant.collection_name,
                assistant_instructions=assistant.instructions  # Pass assistant instructions
            )
            Message.objects.create(thread=thread, role='assistant', content=answer)
            run.status = 'completed'
            run.completed_at = datetime.now()
            run.save()
            logger.info(f"Run {run.id} completed successfully")
        except Exception as e:
            run.status = 'failed'
            run.save()
            Message.objects.create(
                thread=thread,
                role='assistant',
                content=f'Run failed due to an error: {str(e)}'
            )
            logger.error(f"Run {run.id} processing failed: {e}")

class RunListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = RunSerializer
    def get_queryset(self):
        thread_id = self.request.query_params.get('thread_id')
        queryset = Run.objects.filter(thread__tenant=self.user.tenant)
        if thread_id:
            thread = get_object_or_404(Thread, id=thread_id, tenant=self.user.tenant)
            queryset = queryset.filter(thread=thread)
        return queryset

class RunRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = RunSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return Run.objects.filter(thread__tenant=self.user.tenant)


class DocumentAlertCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = DocumentAlertSerializer
    def perform_create(self, serializer):
        document_id = self.request.data.get('document_id')
        document = get_object_or_404(Document, id=document_id, tenant=self.user.tenant)
        serializer.save(document=document)

class DocumentAlertListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = DocumentAlertSerializer
    def get_queryset(self):
        document_id = self.request.query_params.get('document_id')
        queryset = DocumentAlert.objects.filter(document__tenant=self.user.tenant)
        if document_id:
            document = get_object_or_404(Document, id=document_id, tenant=self.user.tenant)
            queryset = queryset.filter(document=document)
        return queryset

class DocumentAlertRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = DocumentAlertSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return DocumentAlert.objects.filter(document__tenant=self.user.tenant)

class DocumentAccessCreateAPIView(TokenAuthenticatedMixin, generics.CreateAPIView):
    serializer_class = DocumentAccessSerializer

    def perform_create(self, serializer):
        try:
            serializer.is_valid(raise_exception=True)
            vector_store = serializer.validated_data['vector_store']
            documents = serializer.validated_data['documents']
            granted_by = self.user

            # Create DocumentAccess records for each document
            created_access = []
            for document in documents:
                access = DocumentAccess(
                    document=document,
                    vector_store=vector_store,
                    granted_by=granted_by
                )
                access.save()
                created_access.append({
                    "document_id": str(document.id),
                    "vector_store_id": str(vector_store.id),
                    "id": access.id
                })

            logger.info(f"Granted access to {len(created_access)} documents for vector_store {vector_store.id}")
            return Response({
                "message": f"Access granted to {len(created_access)} documents.",
                "created_access": created_access
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error granting document access: {e}")
            raise ValidationError({"error": str(e)})

class DocumentAccessRemoveAPIView(TokenAuthenticatedMixin, generics.UpdateAPIView):
    serializer_class = DocumentAccessRemoveSerializer

    def put(self, request, *args, **kwargs):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            vector_store = serializer.validated_data['vector_store']
            valid_document_ids = serializer.validated_data['valid_document_ids']

            # Find and delete DocumentAccess records
            deleted_count = 0
            deleted_ids = []
            for doc_id in valid_document_ids:
                access_records = DocumentAccess.objects.filter(
                    vector_store=vector_store,
                    document__id=doc_id,
                    document__tenant=self.user.tenant
                )
                count = access_records.count()
                if count > 0:
                    deleted_ids.append(doc_id)
                    access_records.delete()
                    deleted_count += count

            if deleted_count == 0:
                logger.info(f"No access records found to remove for vector_store {vector_store.id}")
                return Response({
                    "message": "No access records found for the specified documents."
                }, status=status.HTTP_200_OK)

            logger.info(f"Removed access for {deleted_count} documents from vector_store {vector_store.id}")
            return Response({
                "message": f"Access removed for {deleted_count} documents.",
                "removed_document_ids": deleted_ids
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error removing document access: {e}")
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class DocumentAccessListAPIView(TokenAuthenticatedMixin, generics.ListAPIView):
    serializer_class = DocumentAccessSerializer
    def get_queryset(self):
        return DocumentAccess.objects.filter(granted_by=self.user)

class DocumentAccessRetrieveUpdateDestroyAPIView(TokenAuthenticatedMixin, generics.RetrieveUpdateDestroyAPIView):
    serializer_class = DocumentAccessSerializer
    lookup_field = 'id'
    def get_queryset(self):
        return DocumentAccess.objects.filter(granted_by=self.user)