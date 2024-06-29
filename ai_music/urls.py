from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name=''),  # Redirects to the index view function
    path('convert/', views.image_to_music, name='convert'),
    path('recommend/', views.recommend_music, name='recommend'),
]
