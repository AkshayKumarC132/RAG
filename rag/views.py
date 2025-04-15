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
from .utils import process_file

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

class IngestAPIView(generics.CreateAPIView):
    serializer_class = IngestDocumentSerializer
    # parser_classes = [MultiPartParser, FormParser]  # Allow file uploads

    def create(self, request, token, *args, **kwargs):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user

        # Check for file presence in request
        file = request.FILES.get('file', None)

        if not file:
            return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            # Get tenant from the user's token
            tenant = user.tenant  # Assuming each user has one tenant

            # Process the file and extract text
            text = process_file(file)
            if not text:
                return Response({"error": "File is empty or not supported."}, status=status.HTTP_400_BAD_REQUEST)

            # Save the document in the Document model
            document = Document(
                tenant=tenant,
                title=data['title'],
                content=text
            )
            document.save()

            ingest_document(text, tenant.id, data['title'])
            return Response({"status": "success", "message": "Document ingested."}, status=status.HTTP_201_CREATED)
        except Tenant.DoesNotExist:
            return Response({"error": "Tenant not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AskAPIView(generics.CreateAPIView):
    serializer_class = AskQuestionSerializer

    def create(self, request, token, *args, **kwargs):
        auth_token = get_object_or_404(AuthToken, token_key=token)
        user = auth_token.user

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        try:
            tenant = user.tenant  # Assuming each user has one tenant
            answer = rag_answer(data['question'], tenant.id)
            return Response({"answer": answer}, status=status.HTTP_200_OK)

        except Tenant.DoesNotExist:
            return Response({"error": "Tenant not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)