from django.shortcuts import render
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re
import plotly.graph_objects as go
import numpy as np
from functools import lru_cache
import plotly.express as px
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import os
from django.shortcuts import render, redirect

# --- FUNCIONES AUXILIARES ---
def obtener_imagen():
    """Guarda el gráfico actual en memoria y lo convierte a Base64 para la web."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    imagen = buffer.getvalue()
    buffer.close()
    plt.close('all') # Limpia la memoria para el siguiente gráfico
    return base64.b64encode(imagen).decode('utf-8')

def extraer_pais_limpio(texto):
    if not isinstance(texto, str) or texto.strip() in ('nan', ''):
        return 'Desconocido'

    # Tomar el segmento final
    t = texto.split(',')[-1].strip()

    # Eliminar años al inicio
    t = re.sub(r'^\d{4}\s+', '', t)

    # Eliminar sufijos descriptivos de terremotos
    t = re.sub(r'\s+(Earthquake.*|Aftershock|Foreshock|Sequence)', '', t, flags=re.IGNORECASE)
    t = re.sub(r'[\(\)]', '', t).strip()

    # Estados y territorios de Estados Unidos
    estados_usa = {
        'CA', 'AK', 'NV', 'TX', 'OR', 'WA', 'UT', 'ID', 'MT', 'NM', 'AZ', 'OK', 'WY', 'CO', 'NE', 'OH', 'ME', 'KY', 'NY', 'IL', 'VA', 'NC', 'NH', 'LA',
        'California', 'Alaska', 'Hawaii', 'Nevada', 'Texas', 'Oregon', 'Washington', 'Utah', 'Idaho', 'Montana', 'New Mexico', 'Arizona', 'Oklahoma',
        'Wyoming', 'Colorado', 'Nebraska', 'Ohio', 'Maine', 'Kentucky', 'New York', 'Illinois', 'Virginia', 'North Carolina', 'New Hampshire', 'Louisiana',
        'Northern California', 'Southern Alaska', 'Central Alaska', 'Alaska Peninsula', 'Lake Almanor', 'Ridgecrest', 'Western Montana'
    }
    for est in estados_usa:
        if re.search(r'\b' + re.escape(est) + r'\b', t, flags=re.IGNORECASE):
            return 'United States'

    # Diccionario de países y territorios soberanos
    mapeo_paises = {
        'Chile': ['Chile', 'Chilean', 'Valdivia', 'Juan Fernandez'],
        'Indonesia': ['Indonesia', 'Sumatra', 'Java', 'Banda Sea', 'Molucca', 'Flores Sea', 'Sulawesi', 'Ceram'],
        'Japan': ['Japan', 'Hokkaido', 'Honshu', 'Tokachi', 'Toankai', 'Hyuganada', 'Ryukyu', 'Izu', 'Bonin', 'Volcano Islands'],
        'Papua New Guinea': ['Papua New Guinea', 'New Britain', 'New Ireland', 'Bougainville', 'Bismarck'],
        'New Zealand': ['New Zealand', 'Kermadec'],
        'Fiji': ['Fiji'],
        'Vanuatu': ['Vanuatu', 'Santa Cruz', 'Loyalty Islands'],
        'Solomon Islands': ['Solomon Islands'],
        'Tonga': ['Tonga'],
        'Samoa': ['Samoa'],
        'Philippines': ['Philippines', 'Philippine'],
        'Mexico': ['Mexico', 'MX', 'Baja California', 'Guerrero', 'Oaxaca', 'Chiapas', 'Michoacan', 'Revilla Gigedo'],
        'Peru': ['Peru'],
        'Russia': ['Russia', 'Kuril', 'Okhotsk', 'Severnaya Zemlya', 'Franz Josef', 'Laptev Sea', 'Kamchatka'],
        'China': ['China', 'Xizang', 'Tibet', 'Tibetan', 'Xinjiang', 'Qinghai', 'Sichuan', 'Yunnan'],
        'Taiwan': ['Taiwan'],
        'Ecuador': ['Ecuador', 'Galapagos'],
        'Colombia': ['Colombia'],
        'Argentina': ['Argentina'],
        'Costa Rica': ['Costa Rica'],
        'Guatemala': ['Guatemala'],
        'El Salvador': ['El Salvador'],
        'Nicaragua': ['Nicaragua'],
        'Panama': ['Panama'],
        'Haiti': ['Haiti'],
        'Dominican Republic': ['Dominican Republic'],
        'Puerto Rico': ['Puerto Rico'],
        'Iran': ['Iran'],
        'Turkey': ['Turkey', 'Kahramanmaras'],
        'Greece': ['Greece'],
        'Italy': ['Italy'],
        'Afghanistan': ['Afghanistan'],
        'Pakistan': ['Pakistan'],
        'India': ['India', 'Assam', 'Andaman'],
        'Australia': ['Australia', 'Macquarie'],
        'Canada': ['Canada', 'Haida Gwaii', 'Beaufort Sea', 'Baffin Bay', 'Davis Strait'],
        'Greenland': ['Greenland'],
        'Iceland': ['Iceland', 'Reykjanes'],
        'South Sandwich Islands': ['South Sandwich', 'South Georgia'],
        'Antarctica': ['Antarctica', 'Balleny', 'Ross Sea', 'Drake Passage', 'South Shetland', 'Scotia Sea']
    }

    # Buscar coincidencia en el texto
    for pais, palabras_clave in mapeo_paises.items():
        for kw in palabras_clave:
            if re.search(r'\b' + re.escape(kw) + r'\b', t, flags=re.IGNORECASE):
                return pais

    # Sismos en dorsales / límites oceánicos
    dorsales = ['Ridge', 'Rise', 'Ocean', 'Trench', 'Fracture Zone', 'Triple Junction']
    if any(d in t for d in dorsales):
        return 'Océano / Dorsal Oceánica'

    # Limpieza final de palabras relativas sobrantes
    t = re.sub(r'^(near the coast of|off the coast of|south of the|north of the|east of the|west of the|south of|north of|east of|west of|central|northern|southern|eastern|western)\s+', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+region$', '', t, flags=re.IGNORECASE).strip()

    return t

@lru_cache(maxsize=1)
def cargar_datos():
    """Carga y prepara el dataset de sismos."""
    url = 'https://raw.githubusercontent.com/mcluunaa/Dataset-Visualization/main/sismos_1926_2026_mag5.csv'
    df = pd.read_csv(url)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    if 'año' not in df.columns and 'time' in df.columns:
        df['año'] = df['time'].dt.year

    if 'place' in df.columns:
        df['pais'] = df['place'].apply(extraer_pais_limpio)
    else:
        df['pais'] = 'Desconocido'
    return df

def vista_datos_principales(request):
    df = cargar_datos()
    cols = ['time', 'latitude', 'longitude', 'depth', 'mag', 'pais']
    cols_existentes = [c for c in cols if c in df.columns]
    
    explicaciones = [
        ('time', 'Fecha y hora exacta en que ocurrió el evento sísmico (en formato UTC).'),
        ('latitude', 'Latitud geográfica del hipocentro en grados decimales.'),
        ('longitude', 'Longitud geográfica del hipocentro en grados decimales.'),
        ('depth', 'Profundidad del sismo medida en kilómetros bajo la superficie terrestre.'),
        ('mag', 'Magnitud del evento (medida principal en escala Richter o de momento).'),
        ('pais', 'País o región geográfica inferida a partir de la descripción de ubicación.')
    ]
    
    tabla_muestra = df[cols_existentes].head(15).to_html(classes="table table-striped table-hover", index=False)
    
    return render(request, 'dashboard/graficos.html', {
        'titulo': 'Listado 1: Columnas Principales del Dataset',
        'explicaciones': explicaciones,
        'tabla_datos': tabla_muestra
    })

def vista_datos_secundarios(request):
    df = cargar_datos()
    cols = ['nst', 'rms', 'gap', 'dmin', 'magType']
    cols_existentes = [c for c in cols if c in df.columns]
    
    explicaciones = [
        ('nst', 'Número total de estaciones/sensores sísmicos que registraron el evento.'),
        ('rms', 'Error cuadrático medio del tiempo de llegada de la onda sísmica (precisión del ajuste).'),
        ('gap', 'Brecha azimutal entre estaciones (en grados). Un menor valor indica mejor cobertura.'),
        ('dmin', 'Distancia horizontal desde el epicentro hasta la estación sísmica más cercana.'),
        ('magType', 'Algoritmo o método de cálculo de la magnitud (ej. Mw, Mb, Ms).')
    ]
    
    tabla_muestra = df[cols_existentes].head(15).to_html(classes="table table-striped table-hover", index=False)
    
    return render(request, 'dashboard/graficos.html', {
        'titulo': 'Listado 2: Columnas Relevantes / Secundarias',
        'explicaciones': explicaciones,
        'tabla_datos': tabla_muestra
    })

def vista_datos_especificos(request):
    df = cargar_datos()
    cols = ['id', 'updated', 'type', 'status', 'locationSource', 'magSource']
    cols_existentes = [c for c in cols if c in df.columns]
    
    explicaciones = [
        ('id', 'Identificador alfa-numérico único asignado por el catálogo sísmico USGS.'),
        ('updated', 'Fecha y hora de la última modificación o recálculo del registro.'),
        ('type', 'Clasificación del evento telúrico (ej. earthquake, explosion, quarry blast).'),
        ('status', 'Estado de revisión del dato (ej. reviewed por un sismólogo o automatic).'),
        ('locationSource', 'Sigla de la red o centro sismológico que determinó las coordenadas.'),
        ('magSource', 'Sigla de la institución que calculó el valor de la magnitud.')
    ]
    
    tabla_muestra = df[cols_existentes].head(15).to_html(classes="table table-striped table-hover", index=False)
    
    return render(request, 'dashboard/graficos.html', {
        'titulo': 'Listado 3: Información Específica y Metadatos',
        'explicaciones': explicaciones,
        'tabla_datos': tabla_muestra
    })

JSON_DESC_PATH = os.path.join(os.path.dirname(__file__), 'descripciones.json')

def obtener_descripcion(clave, texto_defecto=""):
    """
    Lee la descripción desde el archivo JSON si existe. 
    Si no existe, devuelve el texto por defecto. (Llamada de 1 sola línea)
    """
    if os.path.exists(JSON_DESC_PATH):
        try:
            with open(JSON_DESC_PATH, 'r', encoding='utf-8') as f:
                descripciones = json.load(f)
                return descripciones.get(clave, texto_defecto)
        except Exception:
            return texto_defecto
    return texto_defecto

def guardar_descripcion_view(request):
    """Procesa el formulario de edición enviado desde la página web."""
    if request.method == 'POST':
        clave = request.POST.get('clave')
        nueva_desc = request.POST.get('descripcion', '').strip()
        
        descripciones = {}
        if os.path.exists(JSON_DESC_PATH):
            try:
                with open(JSON_DESC_PATH, 'r', encoding='utf-8') as f:
                    descripciones = json.load(f)
            except Exception:
                descripciones = {}
                
        descripciones[clave] = nueva_desc
        
        with open(JSON_DESC_PATH, 'w', encoding='utf-8') as f:
            json.dump(descripciones, f, ensure_ascii=False, indent=4)
            
    # Redirige de vuelta a la misma página del gráfico
    return redirect(request.META.get('HTTP_REFERER', '/'))

# --- VISTAS DEL MENÚ ---
def vista_grafico_1(request):
    df = cargar_datos()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5.5))

    sismos_por_año = df.groupby('año').size()
    sns.lineplot(
        x=sismos_por_año.index,
        y=sismos_por_año.values,
        color="#0f4c5c",
        linewidth=2.5,
        label='Sismos registrados',
        ax=ax
    )

    ax.axvspan(1970, 2026, color='#5bc0be', alpha=0.18, label='Período de alta cobertura (red global)')
    ax.axvline(x=1970, color='#2b2d42', linestyle='--', linewidth=1.5, alpha=0.7)

    año_peak = sismos_por_año.idxmax()
    valor_peak = sismos_por_año.max()

    ax.scatter(año_peak, valor_peak, color='#e76f51', s=90, zorder=5, label=f'Peak: {año_peak} ({valor_peak} sismos)')
    ax.annotate(
        f'Peak ({año_peak})\n{valor_peak} eventos',
        xy=(año_peak, valor_peak),
        xytext=(año_peak - 8, valor_peak + 70),
        arrowprops=dict(arrowstyle="->", color='#e76f51', lw=1.2),
        fontweight='bold',
        fontsize=10,
        color='#e76f51'
    )

    ax.set_title('Evolución Histórica de Sismos Registrados por Año (1926-2026)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Año', fontsize=11, labelpad=10)
    ax.set_ylabel('Cantidad de Registros', fontsize=11, labelpad=10)
    ax.set_ylim(0, valor_peak * 1.15)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    fig.tight_layout()

    img = obtener_imagen()

    return render(request, 'dashboard/graficos.html', {
        'titulo': 'Evolución Histórica de Sismos (1926-2026)',
        'clave': 'grafico_1',  # Identificador para la edición
        'descripcion': obtener_descripcion('grafico_1', 'Muestra el número total de eventos registrados por año...'),
        'graficos': [img]
    })

def vista_grafico_2(request):
    # 1. Definir los datos (Usando tus listas exactas)
    paises = ['Indonesia', 'Japón', 'Papúa Nueva Guinea', 'Océanos/Dorsales',
              'Rusia', 'Filipinas', 'Chile', 'Estados Unidos', 'México', 'Nueva Zelanda']
    cantidad = [12564, 7906, 6427, 6330, 6081, 5500, 4800, 3200, 2900, 2100]

    # Crear un DataFrame y ordenarlo de menor a mayor para el gráfico
    df_paises = pd.DataFrame({'País / Región': paises, 'Cantidad': cantidad})
    df_paises = df_paises.sort_values(by='Cantidad', ascending=True)

    # 2. Crear gráfico de barras horizontales interactivo
    fig2 = px.bar(
        df_paises,
        x='Cantidad',
        y='País / Región',
        orientation='h',
        title='<b>Top 10 Regiones con Mayor Cantidad de Sismos</b><br><sup>Magnitud >= 5</sup>',
        text='Cantidad',
        color='Cantidad',
        color_continuous_scale='Viridis'
    )

    # 3. Personalización estética
    fig2.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig2.update_layout(
        template='plotly_white',
        title_font_size=18,
        xaxis_title='Cantidad de Sismos Registrados',
        yaxis_title='',
        coloraxis_showscale=False, # Ocultar barra de colores
        xaxis=dict(showgrid=True, griddash='dot'),
        margin=dict(l=20, r=20, t=60, b=20) # Márgenes para que no se corte en la web
    )

    # 4. Transformar el gráfico de Plotly a HTML (Reemplaza a fig2.show())
    plot_html = fig2.to_html(full_html=False, include_plotlyjs='cdn')

    return render(request, 'dashboard/graficos.html', {
        'titulo': 'Top 10 Regiones con Mayor Sismicidad',
        'clave': 'grafico_2',
        'descripcion': obtener_descripcion('grafico_2', 'Gráfico de barras interactivo que clasifica las regiones...'),
        'plot_html': plot_html
    })

def vista_grafico_3(request):
    # 1. Generar la lista de años y simular datos consistentes con la tendencia
    np.random.seed(42)
    anos = np.arange(1926, 2027)
    
    # Se agrega el asterisco (*) para la multiplicación
    tendencia = 800 + 5 * (anos - 1926) + np.random.normal(0, 150, len(anos))
    tendencia = np.clip(tendencia, 400, 2500)

    # Crear un DataFrame para Plotly
    df_temporal = pd.DataFrame({'Año': anos, 'Cantidad de Sismos': tendencia})

    # 2. Crear gráfico de líneas interactivo con Plotly Express
    fig3 = px.line(
        df_temporal,
        x='Año',
        y='Cantidad de Sismos',
        title='<b>Evolución Temporal de Sismos (Magnitud >= 5)</b><br><sup>Período 1926 - 2026</sup>',
        markers=False,
        color_discrete_sequence=['#0f4c5c']
    )

    # 3. Personalización estética e interactiva
    fig3.update_traces(line=dict(width=2.5))
    fig3.update_layout(
        template='plotly_white',
        title_font_size=18,
        xaxis_title='Año',
        yaxis_title='Cantidad de Sismos Registrados (Aprox)',
        hovermode='x unified',
        xaxis=dict(showgrid=True, griddash='dot'),
        yaxis=dict(showgrid=True, griddash='dot'),
        margin=dict(l=20, r=20, t=60, b=20) # Márgenes adaptados para la web
    )

    # Habilitar zoom y barra de desplazamiento temporal
    fig3.update_xaxes(rangeslider_visible=True)

    # 4. Transformar el gráfico de Plotly a HTML
    plot_html = fig3.to_html(full_html=False, include_plotlyjs='cdn')

    return render(request, 'dashboard/graficos.html', {
        'titulo': 'Evolución Temporal Interactiva',
        'clave': 'grafico_3',
        'descripcion': obtener_descripcion('grafico_3', 'Visualización lineal dinámica equipada con un selector de rango...'),
        'plot_html': plot_html
    })

def vista_grafico_4(request):
    df = cargar_datos()

    # --- Gráfico Nivel Pro: Scatterplot de Magnitud vs Profundidad ---
    sns.set_theme(style="darkgrid", context="talk")
    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=df, 
        x='mag', 
        y='depth', 
        hue='mag', 
        size='mag', 
        sizes=(20, 250), 
        palette='flare', 
        alpha=0.6
    )
    
    # Invertir eje Y para que la profundidad vaya hacia abajo
    plt.gca().invert_yaxis()
    
    plt.title('Relación Sismológica: Profundidad vs Magnitud', pad=15)
    plt.xlabel('Magnitud')
    plt.ylabel('Profundidad (km)')
    plt.legend(title='Magnitud', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()

    # Convertir a imagen para la web
    img = obtener_imagen()

    # Restaurar estilo normal de seaborn para no afectar a los demás gráficos
    sns.set_theme(style="whitegrid")
    
    return render(request, 'dashboard/graficos.html', {
        'titulo': 'Análisis Avanzado: Profundidad vs Magnitud',
        'clave': 'grafico_4',
        'descripcion': obtener_descripcion('grafico_4', 'Diagrama de dispersión que analiza la distribución física...'),
        'graficos': [img]
    })

def mapa_terremotos_view(request):
    # Carga tu dataset con la columna 'pais' ya limpia
    df = cargar_datos()

    # 1. Filtramos sismos mayores a 7.0 y ordenamos por magnitud ascendente
    df_filtrado = df[df['mag'] > 7.0].copy()
    df_filtrado['fecha_str'] = pd.to_datetime(df_filtrado['time']).dt.strftime('%Y-%m-%d')
    df_filtrado = df_filtrado.sort_values('mag', ascending=True)

    # 2. Escala de tamaño para los círculos (fórmula corregida)
    mag_min, mag_max = 7.0, 9.5
    df_filtrado['size_scaled'] = 4 + 22 * ((df_filtrado['mag'] - mag_min) / (mag_max - mag_min)) ** 2.2

    # 3. Paleta continua cálida
    paleta_calida = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026', '#1a0000']

    # 4. Construcción del gráfico
    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=df_filtrado['latitude'],
            lon=df_filtrado['longitude'],
            mode='markers',
            marker=dict(
                size=df_filtrado['size_scaled'],
                color=df_filtrado['mag'],
                colorscale=paleta_calida,
                cmin=7.0,
                cmax=9.5,
                showscale=True,
                colorbar=dict(
                    title=dict(text="<b>Magnitud</b>", side="top"),
                    ticks="outside",
                    thickness=16,
                    len=0.8
                ),
                line=dict(width=0.6, color='black'),
                opacity=0.9
            ),
            text=df_filtrado['pais'],
            customdata=np.stack((df_filtrado['fecha_str'], df_filtrado['mag'], df_filtrado['depth']), axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br><br>" +
                "<b>Fecha:</b> %{customdata[0]}<br>" +
                "<b>Magnitud:</b> %{customdata[1]:.2f}<br>" +
                "<b>Profundidad:</b> %{customdata[2]:.1f} km" +
                "<extra></extra>"
            )
        )
    )

    # 5. Proyección ortográfica
    fig.update_geos(
        projection_type="orthographic",
        showcoastlines=True,
        coastlinecolor="#b0b0b0",
        showland=True,
        landcolor="#f6f6f4",
        showocean=True,
        oceancolor="#eaf2f8",
        showcountries=True,
        countrycolor="#dcdcdc"
    )

    # 6. Configuración de dimensiones y centrado
    fig.update_layout(
        height=700,
        title={
            'text': "<b>Grandes Terremotos Globales (Magnitud > 7.0)</b>",
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.96,
            'font': {'size': 18}
        },
        margin=dict(l=10, r=10, t=50, b=10)
    )

    # 7. Convertimos la figura a fragmento HTML ejecutable
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    context = {
        'titulo': 'Mapa Interactivo 3D',
        'plot_html': plot_html
    }
    return render(request, 'dashboard/graficos.html', {
        'titulo': 'Mapa Interactivo 3D',
        'clave': 'grafico_5',
        'descripcion': obtener_descripcion('grafico_5', 'Globo terráqueo tridimensional interactivo...'),
        'plot_html': plot_html
    })