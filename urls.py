from django.contrib import admin
from django.urls import path
from django.urls import include
from . import views

from sdfg. improts sdfgsdfg

urlpatterns = [
    path('', views.home, name = 'myblog-home'),
    path('rango', include('rango.urls')),
    path('about/', views.about, name = 'myblog-about'),
]
