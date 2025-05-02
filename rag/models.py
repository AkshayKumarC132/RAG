# rag/models.py

from django.db import models
from django.contrib.auth.models import AbstractUser

class Tenant(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

class User(AbstractUser):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="users")

class Document(models.Model):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    vector_id = models.CharField(max_length=36, unique=True)  # Store UUID for vector DB
    summary = models.TextField(blank=True, null=True)
    keywords = models.JSONField(blank=True, null=True)  # for array of keywords
    
    def __str__(self):
        return self.title
    

class ChatHistory(models.Model):
    vector_id = models.CharField(max_length=255, unique=True)
    history = models.JSONField(default=list)  # Store as JSON array
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.vector_id

class DocumentAlert(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='alerts')
    keyword = models.CharField(max_length=255)
    snippet = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Alert on {self.document.title}: {self.keyword}"


class MultiFileChatSession(models.Model):
    session_id = models.CharField(unique=True, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    vector_ids = models.JSONField(default=list)  # list of vector_id strings
    vector_hash = models.CharField(max_length=128, null=True, blank=True)  # 👈 add this
    history = models.JSONField(default=list)     # [{"question": "...", "answer": "..."}]
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Session {self.session_id}"