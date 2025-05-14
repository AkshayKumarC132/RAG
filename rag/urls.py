#  rag/urls.py

from django.urls import path
from .views import (
    RegisterAPI, LoginView, LogoutView, ProtectedView,
    TenantCreateAPIView, TenantListAPIView, TenantRetrieveUpdateDestroyAPIView,
    UserListAPIView, UserRetrieveUpdateDestroyAPIView,
    VectorStoreCreateAPIView, VectorStoreListAPIView, VectorStoreRetrieveUpdateDestroyAPIView,
    IngestAPIView, DocumentListAPIView, DocumentRetrieveUpdateDestroyAPIView,
    AssistantCreateAPIView, AssistantListAPIView, AssistantRetrieveUpdateDestroyAPIView,
    ThreadCreateAPIView, ThreadListAPIView, ThreadRetrieveUpdateDestroyAPIView, ThreadMessagesAPIView,
    MessageCreateAPIView, MessageListAPIView, MessageRetrieveUpdateDestroyAPIView,
    RunCreateAPIView, RunListAPIView, RunRetrieveUpdateDestroyAPIView,
    DocumentAccessCreateAPIView, DocumentAccessListAPIView, DocumentAccessRetrieveUpdateDestroyAPIView,
    DocumentAlertCreateAPIView, DocumentAlertListAPIView, DocumentAlertRetrieveUpdateDestroyAPIView,
    OpenAIKeyCreateAPIView, OpenAIKeyListAPIView, OpenAIKeyRetrieveUpdateDestroyAPIView,
)

urlpatterns = [
    # Authentication
    path('register/', RegisterAPI.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/<str:token>/', LogoutView.as_view(), name='logout'),
    path('protected/<str:token>/', ProtectedView.as_view(), name='protected'),

    # Tenant
    path('tenant/', TenantCreateAPIView.as_view(), name='tenant-create'),
    path('tenant/list/', TenantListAPIView.as_view(), name='tenant-list'),
    path('tenant/<int:id>/', TenantRetrieveUpdateDestroyAPIView.as_view(), name='tenant-detail'),

    # User
    path('user/<str:token>/list/', UserListAPIView.as_view(), name='user-list'),
    path('user/<str:token>/<int:id>/', UserRetrieveUpdateDestroyAPIView.as_view(), name='user-detail'),

    # Vector Store
    path('vector-store/<str:token>/', VectorStoreCreateAPIView.as_view(), name='vector-store-create'),
    path('vector-store/<str:token>/list/', VectorStoreListAPIView.as_view(), name='vector-store-list'),
    path('vector-store/<str:token>/<str:id>/', VectorStoreRetrieveUpdateDestroyAPIView.as_view(), name='vector-store-detail'),

    # Document
    path('document/<str:token>/ingest/', IngestAPIView.as_view(), name='document-ingest'),
    path('document/<str:token>/list/', DocumentListAPIView.as_view(), name='document-list'),
    path('document/<str:token>/<str:id>/', DocumentRetrieveUpdateDestroyAPIView.as_view(), name='document-detail'),

    # Assistant
    path('assistant/<str:token>/', AssistantCreateAPIView.as_view(), name='assistant-create'),
    path('assistant/<str:token>/list/', AssistantListAPIView.as_view(), name='assistant-list'),
    path('assistant/<str:token>/<str:id>/', AssistantRetrieveUpdateDestroyAPIView.as_view(), name='assistant-detail'),

    # Thread
    path('thread/<str:token>/', ThreadCreateAPIView.as_view(), name='thread-create'),
    path('thread/<str:token>/list/', ThreadListAPIView.as_view(), name='thread-list'),
    path('thread/<str:token>/<str:id>/', ThreadRetrieveUpdateDestroyAPIView.as_view(), name='thread-detail'),
    path('thread/<str:token>/<str:id>/messages/', ThreadMessagesAPIView.as_view(), name='thread-messages'),

    # Message
    path('message/<str:token>/', MessageCreateAPIView.as_view(), name='message-create'),
    path('message/<str:token>/list/', MessageListAPIView.as_view(), name='message-list'),
    path('message/<str:token>/<int:id>/', MessageRetrieveUpdateDestroyAPIView.as_view(), name='message-detail'),

    # Run
    path('run/<str:token>/', RunCreateAPIView.as_view(), name='run-create'),
    path('run/<str:token>/list/', RunListAPIView.as_view(), name='run-list'),
    path('run/<str:token>/<str:id>/', RunRetrieveUpdateDestroyAPIView.as_view(), name='run-detail'),

    # Document Access
    path('document-access/<str:token>/', DocumentAccessCreateAPIView.as_view(), name='document-access-create'),
    path('document-access/<str:token>/list/', DocumentAccessListAPIView.as_view(), name='document-access-list'),
    path('document-access/<str:token>/<int:id>/', DocumentAccessRetrieveUpdateDestroyAPIView.as_view(), name='document-access-detail'),

    # Document Alert
    path('document-alert/<str:token>/', DocumentAlertCreateAPIView.as_view(), name='document-alert-create'),
    path('document-alert/<str:token>/list/', DocumentAlertListAPIView.as_view(), name='document-alert-list'),
    path('document-alert/<str:token>/<int:id>/', DocumentAlertRetrieveUpdateDestroyAPIView.as_view(), name='document-alert-detail'),

    # OpenAI API Key
    path('openai-key/<str:token>/', OpenAIKeyCreateAPIView.as_view(), name='openai-key-create'),
    path('openai-key/<str:token>/list/', OpenAIKeyListAPIView.as_view(), name='openai-key-list'),
    path('openai-key/<str:token>/<int:id>/', OpenAIKeyRetrieveUpdateDestroyAPIView.as_view(), name='openai-key-detail'),

]
