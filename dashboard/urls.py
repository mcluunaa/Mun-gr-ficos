from django.urls import path
from . import views

urlpatterns = [
    # Ruta obligatoria para guardar descripciones editadas desde la web
    path('guardar-descripcion/', views.guardar_descripcion_view, name='guardar_descripcion'),

    # Rutas para el Listado de Datos
    path('datos-principales/', views.vista_datos_principales, name='datos_principales'),
    path('datos-secundarios/', views.vista_datos_secundarios, name='datos_secundarios'),
    path('datos-especificos/', views.vista_datos_especificos, name='datos_especificos'),
    
    # Rutas para los Gráficos
    path('grafico-1/', views.vista_grafico_1, name='grafico_1'),
    path('grafico-2/', views.vista_grafico_2, name='grafico_2'),
    path('grafico-3/', views.vista_grafico_3, name='grafico_3'),
    path('grafico-4/', views.vista_grafico_4, name='grafico_4'),
    path('mapa-terremotos/', views.mapa_terremotos_view, name='mapa_terremotos'),
]