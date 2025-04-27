# rag/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.RegisterAPI.as_view(), name='register'),
    path('login/', views.LoginView.as_view(), name='knox_login'),
    path('logout/<str:token>/', views.LogoutView.as_view(), name='logout'),
    path('protected/<str:token>/', views.ProtectedView.as_view(), name='protected'),
    path('ingest/<str:token>/', views.IngestAPIView.as_view(), name='upload_and_ingest'),
    path('list-documents/<str:token>/', views.IngestAPIView.as_view(), name='list-documents'),
    path('ask/<str:token>/', views.AskAPIView.as_view(), name='ask_question'),
    path('retrieve/document/<str:token>/<vector_id>', views.RetrieveByVectorIdAPIView.as_view(), name='document'),
    path('delete/<str:token>/<vector_id>', views.DeleteDocumentAPIView.as_view(), name='delete_document'),

    path('chat-history/<str:token>/<str:vector_id>/', views.chat_history, name='chat-history'),

    path('multifile-ask/<str:token>/', views.MultiFileAskAPIView.as_view(), name='multifile-ask'),
    path('global-ask/<str:token>/', views.GlobalAskAPIView.as_view(), name='global-ask'),
]
