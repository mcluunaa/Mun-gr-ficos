from django.contrib import admin
from django.urls import path, include  # <-- Es vital agregar 'include' aquí

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('dashboard.urls')),  # <-- Esto conecta tus gráficos al servidor principal
]