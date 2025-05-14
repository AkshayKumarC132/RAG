# rag/models.py


from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid
from django.core.exceptions import ValidationError
from project import settings
import requests

# Define generator functions at module level
def generate_prefixed_uuid_doc():
    return f"doc_{uuid.uuid4()}"

def generate_prefixed_uuid_vs():
    return f"vs_{uuid.uuid4()}"

def generate_prefixed_uuid_run():
    return f"run_{uuid.uuid4()}"

def generate_prefixed_uuid_thread():
    return f"thread_{uuid.uuid4()}"

def generate_prefixed_uuid_assistant():
    return f"assistant_{uuid.uuid4()}"

class Tenant(models.Model):
    name = models.CharField(max_length=255, unique=True)
    collection_name = models.CharField(max_length=255, default="default_collection")

    def __str__(self):
        return self.name

    def clean(self):
        if not self.collection_name:
            raise ValidationError("Collection name cannot be empty.")
        if Tenant.objects.exclude(pk=self.pk).filter(collection_name=self.collection_name).exists():
            raise ValidationError("Collection name must be unique.")

class User(AbstractUser):
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="users")

class OpenAIKey(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="openai_keys")
    api_key = models.CharField(max_length=255)
    is_valid = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"OpenAI Key for {self.user.username}"

    def clean(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        test_payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello, validate my API key."}],
            "max_tokens": 5,
        }
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=test_payload)
            if response.status_code == 200:
                self.is_valid = True
            else:
                self.is_valid = False
                error_detail = response.json().get("error", {}).get("message", response.text)
                raise ValidationError(f"Invalid OpenAI API key or quota exceeded: {error_detail}")
        except requests.exceptions.RequestException as e:
            self.is_valid = False
            raise ValidationError(f"Failed to validate OpenAI API key: {str(e)}")

class VectorStore(models.Model):
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_vs, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="vector_stores")
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"VectorStore {self.id} - {self.name}"

class Document(models.Model):
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_doc, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="documents")
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="documents")
    title = models.CharField(max_length=255)
    content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    summary = models.TextField(blank=True, null=True)
    keywords = models.JSONField(blank=True, null=True)

    def __str__(self):
        return self.title

class Assistant(models.Model):
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_assistant, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="assistants")
    name = models.CharField(max_length=255)
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="assistants", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    creator = models.ForeignKey(User, on_delete=models.CASCADE, related_name="created_assistants", null=True)

    def __str__(self):
        return f"Assistant {self.id} - {self.name}"

class Thread(models.Model):
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_thread, editable=False)
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE, related_name="threads")
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="threads")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Thread {self.id}"

class Message(models.Model):
    id = models.AutoField(primary_key=True)
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=20)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message in {self.thread.id} by {self.role}"

class Run(models.Model):
    id = models.CharField(primary_key=True, max_length=50, default=generate_prefixed_uuid_run, editable=False)
    thread = models.ForeignKey(Thread, on_delete=models.CASCADE, related_name="runs")
    assistant = models.ForeignKey(Assistant, on_delete=models.CASCADE, related_name="runs")
    status = models.CharField(max_length=20, default="queued")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Run {self.id} in {self.thread.id}"

class DocumentAlert(models.Model):
    id = models.AutoField(primary_key=True)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='alerts')
    keyword = models.CharField(max_length=255)
    snippet = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Alert on {self.document.title}: {self.keyword}"

class DocumentAccess(models.Model):
    id = models.AutoField(primary_key=True)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="access")
    vector_store = models.ForeignKey(VectorStore, on_delete=models.CASCADE, related_name="access")
    granted_by = models.ForeignKey(User, on_delete=models.CASCADE)
    granted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('document', 'vector_store')