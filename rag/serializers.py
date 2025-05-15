# rag/serializers.py

from rest_framework import serializers
from .models import Tenant, User, VectorStore, Document, Assistant, Thread, Message, Run, DocumentAccess, OpenAIKey, DocumentAlert
from django.shortcuts import get_object_or_404


class TenantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tenant
        fields = ['id', 'name', 'collection_name']

    def validate_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Tenant name cannot be empty.")
        return value

    def validate_collection_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Collection name cannot be empty.")
        return value

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    tenant = serializers.PrimaryKeyRelatedField(queryset=Tenant.objects.all(), required=False)
    tenant_name = serializers.CharField(write_only=True, required=False)
    collection_name = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'tenant', 'tenant_name', 'collection_name')

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
        collection_name = validated_data.get('collection_name')
        if tenant_name:
            tenant, created = Tenant.objects.get_or_create(name=tenant_name, defaults={'collection_name': collection_name or 'default_collection'})
        user = User.objects.create_user(
            username=validated_data['username'],
            email=validated_data.get('email'),
            password=validated_data['password'],
            tenant=tenant
        )
        return user

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=False)

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'tenant', 'password')
        read_only_fields = ('id',)

    def validate_username(self, value):
        if not value.strip():
            raise serializers.ValidationError("Username cannot be empty.")
        return value

    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField()

    def validate(self, attrs):
        username = attrs.get('username')
        password = attrs.get('password')
        if not username or not password:
            raise serializers.ValidationError("Must include 'username' and 'password'.", code='authorization')
        user = User.objects.filter(username=username).first()
        if not user or not user.check_password(password):
            raise serializers.ValidationError("Invalid credentials.", code='authorization')
        attrs['user'] = user
        return attrs

class VectorStoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = VectorStore
        fields = ['id', 'name', 'created_at']
        read_only_fields = ['id', 'created_at']

    def validate_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Vector store name cannot be empty.")
        return value

class IngestDocumentSerializer(serializers.Serializer):
    file = serializers.FileField(required=False)
    s3_file_url = serializers.URLField(required=False)
    vector_store_id = serializers.CharField(required=True)

    def validate(self, data):
        if not data.get('file') and not data.get('s3_file_url'):
            raise serializers.ValidationError("Either 'file' or 's3_file_url' must be provided.")
        if data.get('file') and data.get('s3_file_url'):
            raise serializers.ValidationError("Provide only one of 'file' or 's3_file_url'.")
        return data

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'vector_store', 'uploaded_at']
        read_only_fields = ['id', 'uploaded_at']

    def validate_title(self, value):
        if not value.strip():
            raise serializers.ValidationError("Document title cannot be empty.")
        return value

class AssistantSerializer(serializers.ModelSerializer):
    vector_store_id = serializers.CharField(write_only=True, required=False)
    instructions = serializers.CharField(required=False, allow_blank=True)

    class Meta:
        model = Assistant
        fields = ['id', 'name', 'vector_store_id', 'created_at','instructions', 'updated_at']
        read_only_fields = ['id', 'created_at','updated_at']

    def validate_name(self, value):
        if not value.strip():
            raise serializers.ValidationError("Assistant name cannot be empty.")
        return value

class ThreadSerializer(serializers.ModelSerializer):
    vector_store_id = serializers.CharField(write_only=True)

    class Meta:
        model = Thread
        fields = ['id', 'vector_store_id', 'created_at']
        read_only_fields = ['id', 'created_at']

    def validate_vector_store_id(self, value):
        try:
            VectorStore.objects.get(id=value)
        except VectorStore.DoesNotExist:
            raise serializers.ValidationError("Invalid vector_store_id.")
        return value

class MessageSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(write_only=True)

    class Meta:
        model = Message
        fields = ['id', 'thread_id', 'role', 'content', 'created_at']
        read_only_fields = ['id', 'role', 'created_at']

    def validate_content(self, value):
        if not value.strip():
            raise serializers.ValidationError("Message content cannot be empty.")
        return value

    def validate_thread_id(self, value):
        try:
            Thread.objects.get(id=value)
        except Thread.DoesNotExist:
            raise serializers.ValidationError("Invalid thread_id.")
        return value

class RunSerializer(serializers.ModelSerializer):
    thread_id = serializers.CharField(write_only=True)
    assistant_id = serializers.CharField(write_only=True)

    class Meta:
        model = Run
        fields = ['id', 'thread_id', 'status', 'assistant_id', 'created_at', 'completed_at']
        read_only_fields = ['id', 'created_at', 'completed_at']

    def validate_thread_id(self, value):
        try:
            Thread.objects.get(id=value)
        except Thread.DoesNotExist:
            raise serializers.ValidationError("Invalid thread_id.")
        return value

    def validate_assistant_id(self, value):
        try:
            Assistant.objects.get(id=value)
        except Assistant.DoesNotExist:
            raise serializers.ValidationError("Invalid assistant_id.")
        return value

# class DocumentAccessSerializer(serializers.ModelSerializer):
#     document_id = serializers.CharField(write_only=True)
#     vector_store_id = serializers.CharField(write_only=True)
#     granted_by = serializers.PrimaryKeyRelatedField(read_only=True)

#     class Meta:
#         model = DocumentAccess
#         fields = ['id', 'document_id', 'vector_store_id', 'granted_by', 'granted_at']
#         read_only_fields = ['id', 'granted_at']
class DocumentAccessSerializer(serializers.ModelSerializer):
    document_ids = serializers.ListField(
        child=serializers.CharField(),
        write_only=True,
        help_text="List of document IDs to grant access to"
    )
    vector_store_id = serializers.CharField(write_only=True)
    granted_by = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = DocumentAccess
        fields = ['id', 'document_ids', 'vector_store_id', 'granted_by', 'granted_at']
        read_only_fields = ['id', 'granted_at']

    def validate(self, data):
        vector_store_id = data.get('vector_store_id')
        document_ids = data.get('document_ids')
        user = self.context['request'].user

        # Validate vector_store_id
        if not vector_store_id:
            raise serializers.ValidationError({"vector_store_id": "This field is required."})
        vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=user.tenant)

        # Validate document_ids
        if not document_ids:
            raise serializers.ValidationError({"document_ids": "At least one document ID is required."})
        invalid_ids = []
        valid_documents = []
        for doc_id in document_ids:
            try:
                doc = Document.objects.get(id=doc_id, tenant=user.tenant)
                valid_documents.append(doc)
            except Document.DoesNotExist:
                invalid_ids.append(doc_id)
        if invalid_ids:
            raise serializers.ValidationError({
                "document_ids": f"Invalid document IDs: {', '.join(invalid_ids)}"
            })

        # Check for existing access to prevent duplicates
        existing_access = DocumentAccess.objects.filter(
            vector_store=vector_store,
            document__in=valid_documents
        ).values_list('document__id', flat=True)
        existing_ids = [str(doc_id) for doc_id in existing_access]
        duplicate_ids = [doc_id for doc_id in document_ids if doc_id in existing_ids]
        if duplicate_ids:
            raise serializers.ValidationError({
                "document_ids": f"Access already granted for document IDs: {', '.join(duplicate_ids)}"
            })

        data['vector_store'] = vector_store
        data['documents'] = valid_documents
        return data

class DocumentAccessRemoveSerializer(serializers.Serializer):
    document_ids = serializers.ListField(
        child=serializers.CharField(),
        write_only=True,
        help_text="List of document IDs to remove access from"
    )
    vector_store_id = serializers.CharField(write_only=True)

    def validate(self, data):
        vector_store_id = data.get('vector_store_id')
        document_ids = data.get('document_ids')
        user = self.context['request'].user

        # Validate vector_store_id
        if not vector_store_id:
            raise serializers.ValidationError({"vector_store_id": "This field is required."})
        vector_store = get_object_or_404(VectorStore, id=vector_store_id, tenant=user.tenant)

        # Validate document_ids
        if not document_ids:
            raise serializers.ValidationError({"document_ids": "At least one document ID is required."})
        invalid_ids = []
        valid_document_ids = []
        for doc_id in document_ids:
            if Document.objects.filter(id=doc_id, tenant=user.tenant).exists():
                valid_document_ids.append(doc_id)
            else:
                invalid_ids.append(doc_id)
        if invalid_ids:
            raise serializers.ValidationError({
                "document_ids": f"Invalid document IDs: {', '.join(invalid_ids)}"
            })

        data['vector_store'] = vector_store
        data['valid_document_ids'] = valid_document_ids
        return data

class OpenAIKeySerializer(serializers.ModelSerializer):
    class Meta:
        model = OpenAIKey
        fields = ['id', 'api_key', 'is_valid', 'created_at', 'updated_at']
        read_only_fields = ['id', 'is_valid', 'created_at', 'updated_at']

    def validate_api_key(self, value):
        if not value.strip():
            raise serializers.ValidationError("API key cannot be empty.")
        return value

class DocumentAlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentAlert
        fields = ['id', 'document', 'keyword', 'snippet', 'created_at']
        read_only_fields = ['id', 'created_at']

    def validate_keyword(self, value):
        if not value.strip():
            raise serializers.ValidationError("Keyword cannot be empty.")
        return value