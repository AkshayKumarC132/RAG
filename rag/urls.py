# rag/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.RegisterAPI.as_view(), name='register'),
    path('login/', views.LoginView.as_view(), name='knox_login'),
    path('logout/<str:token>/', views.LogoutView.as_view(), name='logout'),
    path('protected/<str:token>/', views.ProtectedView.as_view(), name='protected'),
    path('ingest/<str:token>/', views.IngestAPIView.as_view(), name='upload_and_ingest'),
    path('ask/<str:token>/', views.AskAPIView.as_view(), name='ask_question'),
]
