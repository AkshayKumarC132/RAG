# rag/serializers.py

from rest_framework import serializers
from .models import Tenant
from django.contrib.auth import get_user_model
from .models import Document, ChatHistory, MultiFileChatSession, DocumentAccess

User = get_user_model()

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    tenant = serializers.PrimaryKeyRelatedField(
        queryset=Tenant.objects.all(),
        required=False
    )
    tenant_name = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'tenant', 'tenant_name')

    def validate(self, data):
        tenant = data.get('tenant')
        tenant_name = data.get('tenant_name')

        if not tenant and not tenant_name:
            raise serializers.ValidationError("Either 'tenant' or 'tenant_name' must be provided.")

        if tenant and tenant_name:
            raise serializers.ValidationError("Provide only one of 'tenant' or 'tenant_name', not both.")

        return data

    def create(self, validated_data):
        tenant = validated_data.get('tenant')
        tenant_name = validated_data.get('tenant_name')

        if tenant_name:
            tenant, created = Tenant.objects.get_or_create(name=tenant_name)

        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email'),
            password=validated_data['password'],
            tenant=tenant
        )
        return user
    

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'tenant')
        read_only_fields = ('id',)

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

    def validate(self, attrs):
        username = attrs.get('username')
        password = attrs.get('password')

        if username and password:
            user = User.objects.filter(username=username).first()

            if user:
                if not user.check_password(password):
                    msg = 'Unable to log in with provided credentials.'
                    raise serializers.ValidationError(msg, code='authorization')
            else:
                msg = 'Unable to log in with provided credentials.'
                raise serializers.ValidationError(msg, code='authorization')

        else:
            msg = 'Must include "username" and "password".'
            raise serializers.ValidationError(msg, code='authorization')

        attrs['user'] = user
        return attrs


class IngestDocumentSerializer(serializers.Serializer):
    file = serializers.FileField(required=False)
    s3_file_url = serializers.URLField(required=False)

    def validate(self, data):
        if not data.get('file') and not data.get('s3_file_url'):
            raise serializers.ValidationError("Either file or s3_file_url must be provided.")
        return data

class AskQuestionSerializer(serializers.Serializer):
    question = serializers.CharField()
    vector_id = serializers.CharField(required=False)
    chat_history = serializers.JSONField(required=False)
    user_identifier = serializers.CharField(max_length=255, required=True)

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['title', 'vector_id', 'uploaded_at',]

class ChatHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatHistory
        fields = ['vector_id', 'user_identifier', 'history', 'updated_at']

class MultiFileChatSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = MultiFileChatSession
        fields = ['session_id', 'user_identifier', 'vector_ids', 'vector_hash', 'history', 'created_at', 'updated_at']

class DocumentAccessSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentAccess
        fields = ['document', 'user_identifier', 'granted_at']