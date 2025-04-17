# rag/views.py

from django.http import JsonResponse
from .retrival import rag_answer
from .injection_pipeline import ingest_document
from .models import Tenant,Document
import json
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from knox.models import AuthToken
from rest_framework import generics
from .serializers import *
from django.contrib.auth import authenticate, get_user_model
from django.shortcuts import get_object_or_404
from rest_framework.parsers import MultiPartParser, FormParser
from .utils import extract_text_from_file, insert_document_to_vectorstore, ask_question
import tempfile
from pathlib import Path
import time

User = get_user_model()

class ProtectedView(APIView):

    def get(self, request,token=None, format=None):
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
class IngestAPIView(APIView):
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user
        serializer = IngestDocumentSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']
            source_type = "file"  # or dynamically decide if needed

            
            # Save to a temp file
            # with tempfile.NamedTemporaryFile(delete=False) as tmp:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            print("Temporary file created at: ", tmp_path)
            try:
                # Get tenant from the user's token
                tenant = user.tenant  # Assuming each user has one tenant
                print("Tenant ID: ", tenant.id)
                print("File name: ", uploaded_file.name)
                
                extracted_text = extract_text_from_file(tmp_path, uploaded_file.name)
                file_ext = Path(uploaded_file.name).suffix.lower()


                start = time.time()

                insert_document_to_vectorstore(extracted_text, source_type, file_ext)

                end = time.time()
                print(f"[+] Embedding and storing in vector store took {end - start:.2f} seconds")

                # Save the document in the Document model
                document = Document(
                    tenant=tenant,
                    title=uploaded_file.name,
                    content=extracted_text
                )
                document.save()

                return Response({"message": "File ingested and stored successfully.",
                                 "file name" : uploaded_file.name,
                                 "document_id": document.id}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            finally:
                Path(tmp_path).unlink(missing_ok=True)  # Clean up temp file
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AskAPIView(APIView):
    def post(self, request, token):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        # user = auth_token.user
        serializer = AskQuestionSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            source_type = "file"  # or dynamic based on your design
            
            try:
                answer = ask_question(question, source_type)
                return Response({"answer": answer}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)