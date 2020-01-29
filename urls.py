from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name = 'myblog-home'),
    path('about/', views.about, name = 'myblog-about'),
]
