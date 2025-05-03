# rag/serializers.py

from rest_framework import serializers
from .models import Tenant
from django.contrib.auth import get_user_model
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

class AskQuestionSerializer(serializers.Serializer):
    question = serializers.CharField()
    vector_id = serializers.CharField(required=False)