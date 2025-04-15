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
