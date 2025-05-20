# Líbrerias
# ==============================================================================
import os
import io
import streamlit as st
import pandas as pd
from dateutil.relativedelta import *
import seaborn as sns; sns.set_theme()
import numpy as np
import string
import hashlib
import openpyxl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import re
import plotly.express as px
from scipy.spatial.distance import cosine

# Para Clustering
# ==============================================================================
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Modelo para embeddings (compatible con español)
# ==============================================================================
from sentence_transformers import SentenceTransformer, util

# Preprocesado y modelado
# ==============================================================================
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import pickle  # Para guardar y cargar el modelo

# Descargar recursos necesarios de nltk (una sola vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas relativas portables
DATA_PATH = os.path.join(BASE_DIR, "Data", "Data RFI.xlsx")
LOGO_PATH = os.path.join(BASE_DIR, "Imagenes", "Dipro_Logo1.png")
MODEL_PATH = os.path.join(BASE_DIR, "modelo.pkl")

############################################
# Función Principal de la Página
############################################

def createPage():
    
    ############################################
    # Función para extraer las iniciales
    ############################################

    def obtener_iniciales(texto):
        if pd.isna(texto):
            return ''
        palabras = texto.strip().split()
        iniciales = ''.join([palabra[0].upper() for palabra in palabras if palabra])
        return iniciales
    
    # Función para cargar datos desde Excel (usando caché de Streamlit)
    @st.cache_data(show_spinner='Cargando Datos... Espere...', persist=True)
    def load_df():
        df = pd.read_excel(DATA_PATH)
        #df['Fecha/Hora de creación'] = pd.to_datetime(df['Fecha/Hora de creación'])
        # Damos drop a la columnas "Unnamed:5 hasta 9"
        df = df.drop(columns=['Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9'])
        # Eliminar retornos de carro y saltos de línea del dataframe "_x000D_\n"
        df = df.replace(to_replace=r'(\r\n|\r|\n|_x000D_\\n|_x000D_)', value=' ', regex=True)
        # Quitar espacios duplicados
        df = df.applymap(lambda x: ' '.join(str(x).split()) if isinstance(x, str) else x)
        # Aplicar la función a una nueva columna
        df['ID'] = df['Formato'].apply(obtener_iniciales)
        # Si existe la columna "ID", se renombra a "Registro" y se usa como identificador;
        # de lo contrario, se utiliza el índice.
        if 'ID' in df.columns:
            df.rename(columns={'ID': 'Registro'}, inplace=True)
        else:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Registro'}, inplace=True)
        return df

    # Primero cargamos el DataFrame
    try:
        data = load_df()
    except Exception as e:
        st.error("Error al cargar el archivo 'Data RFI.xlsx'.")
        st.error(str(e))
        return
    
    #############################################
    # Función para Diagrama de Pareto (80-20)
    ############################################
   
    def diagrama_pareto(df):
        # Calcular la frecuencia de cada valor en la columna 'Formato'
        conteo = df['Formato'].value_counts().reset_index()
        conteo.columns = ['Formato', 'Frecuencia']

        # Ordenar de mayor a menor frecuencia
        conteo = conteo.sort_values(by='Frecuencia', ascending=False)

        # Calcular la suma acumulada y el porcentaje acumulado
        conteo['Acumulado'] = conteo['Frecuencia'].cumsum()
        total = conteo['Frecuencia'].sum()
        conteo['Porcentaje Acumulado'] = 100 * conteo['Acumulado'] / total

        # Crear la figura
        fig = go.Figure()

        # Barras de frecuencia
        fig.add_trace(go.Bar(
            x=conteo['Formato'],
            y=conteo['Frecuencia'],
            name='Frecuencia',
            text=conteo['Frecuencia'],
            textposition='outside',
        ))

        # Línea de porcentaje acumulado con etiquetas
        fig.add_trace(go.Scatter(
            x=conteo['Formato'],
            y=conteo['Porcentaje Acumulado'],
            name='Porcentaje Acumulado',
            mode='lines+markers+text',                   # <-- agregamos '+text'
            text=conteo['Porcentaje Acumulado'].round(1).astype(str) + '%',
            textposition='top center',
            texttemplate='%{text}',                      # usamos el texto formateado
            textfont=dict(size=12),                      # opcional: ajustar tamaño
            yaxis='y2'
        ))

        # Layout con fondo transparente y sin rejillas
        fig.update_layout(
            title='Diagrama de Pareto',
            height=600,                                  # opcional: ajustar altura
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(
                title='Formato',
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title='Frecuencia',
                showgrid=False,
                zeroline=False
            ),
            yaxis2=dict(
                title='Porcentaje Acumulado (%)',
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=False,
                range=[0, 110]
            )
        )

        return fig

    ###################################
    # Limpieza y Tokenización del Texto
    ###################################

    def limpiar_tokenizar(texto):
        '''
        Esta función limpia y tokeniza el texto en palabras individuales.
        El orden en el que se va limpiando el texto no es arbitrario.
        El listado de signos de puntuación se ha obtenido de: print(string.punctuation)
        y re.escape(string.punctuation)
        '''
        
        # Se convierte todo el texto a minúsculas
        nuevo_texto = texto.lower()
        # Eliminación de páginas web (palabras que empiezan por "http")
        nuevo_texto = re.sub('http\S+', ' ', nuevo_texto)
        # Eliminación de signos de puntuación
        regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
        nuevo_texto = re.sub(regex , ' ', nuevo_texto)
        # Eliminación de números
        nuevo_texto = re.sub("\d+", ' ', nuevo_texto)
        # Eliminación de espacios en blanco múltiples
        nuevo_texto = re.sub("\\s+", ' ', nuevo_texto)
        # Tokenización por palabras individuales
        nuevo_texto = nuevo_texto.split(sep = ' ')
        # Eliminación de tokens con una longitud < 2
        nuevo_texto = [token for token in nuevo_texto if len(token) > 1]
        
        return(nuevo_texto)
    
    ########################
    #Frecuencia de palabras
    ########################

    # --------------------------------------------------
    # 1) Definimos la función de resumen (tal cual antes)
    # --------------------------------------------------
    @st.cache_data
    def resumen_palabras_por_formato(data: pd.DataFrame) -> pd.DataFrame:
        """
        Recibe un DataFrame con columnas 'Pregunta', 'Formato' e 'Registro'.
        Requiere la función limpiar_tokenizar(texto)->List[str].
        Devuelve un DataFrame con:
        Formato, palabras_distintas, palabras_totales, longitud_media, desviacion
        """
        df = data.copy()
        df['tokens'] = df['Pregunta'].apply(limpiar_tokenizar)
        
        tidy = (
            df
            .explode('tokens')
            .rename(columns={'tokens': 'token'})[
                ['Formato', 'Registro', 'token']
            ]
        )
        
        total = (
            tidy
            .groupby('Formato')['token']
            .count()
            .rename('palabras_totales')
        )
        distintos = (
            tidy
            .groupby('Formato')['token']
            .nunique()
            .rename('palabras_distintas')
        )
        
        longitudes_por_pregunta = (
            tidy
            .groupby(['Formato', 'Registro'])['token']
            .count()
            .reset_index(name='num_tokens')
        )
        stats = (
            longitudes_por_pregunta
            .groupby('Formato')['num_tokens']
            .agg(['mean', 'std'])
            .fillna(0)
            .rename(columns={
                'mean': 'longitud_media',
                'std' : 'desviacion'
            })
        )
        
        resumen = (
            pd.concat([distintos, total, stats], axis=1)
            .reset_index()
        )
        return resumen
    #####################################
    # Palabras más frecuentes por Formato
    #####################################  
    def generar_top_tokens_por_formato(data: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """
        Toma un DataFrame con al menos las columnas 'Pregunta' y 'Formato',
        tokeniza las preguntas, cuenta la frecuencia de cada token por Formato
        y devuelve un DataFrame con los top_n tokens más frecuentes para cada Formato.
        """
        # 1) Tokenización
        data = data.copy()
        data['preguntaRFI_tokenizado'] = data['Pregunta'].apply(limpiar_tokenizar)

        # 2) Unnest / explode
        df_tidy = (
            data
            .explode(column='preguntaRFI_tokenizado')
            .drop(columns='Pregunta')
            .rename(columns={'preguntaRFI_tokenizado': 'token'})
        )

        # 3) Conteo de tokens por Formato
        df_counts = (
            df_tidy
            .groupby(['Formato', 'token'])
            .size()
            .reset_index(name='count')
        )

        # 4) Seleccionar los top_n por Formato
        df_top = (
            df_counts
            .sort_values(['Formato', 'count'], ascending=[True, False])
            .groupby('Formato')
            .head(top_n)
            .reset_index(drop=True)
        )
        return df_top
    
    ############
    # Stopwords 
    ############

    def render_top10_words_by_format(data):
        """
        Render top 10 most frequent tokens per 'Formato' category in a Streamlit app,
        without gridlines and with transparent background.
        Args:
            data (pd.DataFrame): DataFrame with at least 'Pregunta' and 'Formato' columns.
        """
        # Tokenización
        data['preguntaRFI_tokenizado'] = data['Pregunta'].apply(limpiar_tokenizar)

        # Unnest y renombrar
        df_tidy = (
            data
            .explode('preguntaRFI_tokenizado')
            .drop(columns='Pregunta')
            .rename(columns={'preguntaRFI_tokenizado': 'token'})
        )

        # Definir stopwords
        stop_words = set(stopwords.words('spanish'))
        stop_words.update([
            "amp","xa","xe","plano","indica","proyecto","si","apoyo","área","favor",
            "acuerdo","detalle","solicita","rfi","area","existente","buena","oc","cm",
            "aunado","indicar","referente","trabajos","tarde","solicito","cambio","hallazgo",
            "adjunta","producto","nuevo","solicitamos","indiquen","ser","confirmar","embargo",
            "procede","ie","indicarnos","realizar","de","la","se","en","el","cual","debe",
            "quedo","parte"
        ])

        # Filtrar stopwords
        df_tidy = df_tidy[~df_tidy['token'].isin(stop_words)]

        # Preparar gráfico
        formatos = df_tidy['Formato'].unique()
        n_formats = len(formatos)
        fig, axs = plt.subplots(nrows=n_formats, ncols=1, figsize=(12, 4*n_formats))
        if n_formats == 1:
            axs = [axs]

        # Hacer transparente el fondo de la figura
        fig.patch.set_alpha(0.0)

        for ax, formato in zip(axs, formatos):
            df_temp = df_tidy[df_tidy['Formato'] == formato]
            counts = df_temp['token'].value_counts().head(10)

            # Plot sin rejillas
            counts.plot(kind='barh', ax=ax)
            ax.invert_yaxis()
            ax.set_title(formato)

            # Quitar rejillas y hacer fondo transparente
            ax.grid(False)
            ax.set_facecolor('none')

        fig.tight_layout()
        st.pyplot(fig)  

    ############################
    #Función de similitud coseno
    ############################

    def similitud_coseno(a, b):
        return 1 - cosine(a, b) 

    ##################
    #Pivotado cacheado
    ##################

    @st.cache_data(show_spinner=False)
    def pivot_tokens(df: pd.DataFrame) -> pd.DataFrame:
        # Aplica limpieza/tokenización (asume que limpiar_tokenizar ya está definida)
        df = df.copy()
        df['tokens'] = df['Pregunta'].apply(limpiar_tokenizar)
        tidy = (
            df
            .explode('tokens')
            .drop(columns='Pregunta')
            .rename(columns={'tokens': 'token'})
        )
        pivot = (
            tidy
            .groupby(['Formato', 'token'])['token']
            .count()
            .reset_index(name='count')
            .pivot(index='token', columns='Formato', values='count')
        )
        pivot.columns.name = None
        return pivot

    #######################
    #Gráfico de correlación
    #######################

    def plot_correlacion(df: pd.DataFrame):
        pivot = pivot_tokens(df)
        formatos = list(pivot.columns)

        if len(formatos) < 2:
            st.warning("Se requieren al menos 2 formatos distintos para graficar.")
            return

        # Selectboxes para elegir ejes
        col1, col2 = st.columns(2)
        with col1:
            eje_x = st.selectbox("Eje X (Formato)", formatos, index=0)
        with col2:
            eje_y = st.selectbox("Eje Y (Formato)", formatos, index=1)

        # Filtrar y limpiar NA
        temp = pivot[[eje_x, eje_y]].dropna()
        if temp.empty:
            st.warning("No hay datos comunes entre los dos formatos seleccionados.")
            return

        # Calcular coeficiente de correlación por coseno
        coef = similitud_coseno(temp[eje_x].values, temp[eje_y].values)
        st.write(f"**Coeficiente de correlación (coseno):** {coef:.3f}")

        # Gráfico
        fig, ax = plt.subplots(figsize=(12, 10))
        # Log-transform +1 para evitar log(0)
        x_vals = np.log(temp[eje_x] + 1)
        y_vals = np.log(temp[eje_y] + 1)
        sns.regplot(x=x_vals, y=y_vals, scatter_kws={'alpha': 0.05}, ax=ax)

        # Anotaciones en muestra aleatoria de hasta 100 puntos
        n_pts = min(100, len(temp))
        indices = np.random.choice(len(temp), n_pts, replace=False)
        for idx in indices:
            palabra = temp.index[idx]
            ax.annotate(
                palabra,
                xy=(x_vals.iloc[idx], y_vals.iloc[idx]),
                alpha=0.7,
                fontsize=8
            )

        ax.set_xlabel(f"Log({eje_x} + 1)")
        ax.set_ylabel(f"Log({eje_y} + 1)")
        ax.set_title(f"Correlación entre «{eje_x}» y «{eje_y}»")
        ax.grid(False)  # Quitar rejillas

        st.pyplot(fig)

    try:
        st.markdown("<h3 style='text-align: left;'>Análisis de la Data</h3>", unsafe_allow_html=True)
        st.write(""" 
        Pasos a seguir:
        1. Limpiar las columnas relevantes.
        2. Generar embeddings y realizar clustering (solo si hay nuevos datos).
        3. Calcular la frecuencia por grupo y reestructurar el DataFrame final.
        4. Exportar el resultado a Excel y realizar un análisis exploratorio (Pareto).
        """)
        ################
        # Carga de Data
        ################
        st.markdown("<h3 style='text-align: left;'>Data</h3>", unsafe_allow_html=True)
        st.dataframe(data)
        
        #############################
        # Cuantificar Áreas (Formato)
        #############################

        st.markdown("<h3 style='text-align: left;'>Cuantificar Áreas (Formato)</h3>", unsafe_allow_html=True)
        # Obtenemos la lista de formatos y eliminamos el último elemento (si lo necesitas)
        lista_areas = data['Formato'].unique().tolist()
        lista_areas.pop(-1)
        # Cantidad de áreas
        cantidad_areas = len(lista_areas)
        # Convertimos la lista a una cadena separada por comas
        areas_str = ", ".join(map(str, lista_areas))
        st.write(
            f"La cantidad de áreas (Formato) presentes en el documento son: {cantidad_areas}. "
            f"Las áreas (Formato) presentes son: {areas_str}."
        )

        ######################
        # Diagrama de Pareto
        ######################

        st.markdown("<h3>Gráfico Pareto</h3>", unsafe_allow_html=True)
        fig_pareto = diagrama_pareto(data)
        st.plotly_chart(fig_pareto, use_container_width=True)

        ########################
        # Frecuencia de Palabras
        ########################

        df_resumen = resumen_palabras_por_formato(data)
    
        # Mostrar tabla
        st.subheader("Comprender las Preguntas del RFI")
        st.dataframe(df_resumen, use_container_width=True)
        
    
        st.subheader("Gráfica comparativa")
        # Tu DataFrame ya preparado:
        chart = (
            df_resumen
            .set_index('Formato')[['palabras_totales','palabras_distintas']]
            .sort_values('palabras_totales', ascending=False)
            .reset_index()
        )

        # Creamos el bar chart con altura personalizada:
        fig = px.bar(
            chart,
            x='Formato',
            y=['palabras_totales','palabras_distintas'],
            barmode='group',
            height=700,                # aquí ajustas la altura en píxeles
            labels={
                'value': 'Conteo',
                'variable': 'Métrica',
                'Formato': 'Formato'
            }
        )

        st.plotly_chart(fig, use_container_width=True)
        st.write("El tipo de pregunta FDI en los formatos de Bodega Aurrera y Walmart Supercenter similares en cuanto a longitud media y desviación. STD = 0 ⇾ no hay dispersión en los datos que estás agrupando (uno o todos iguales).")
        
        #######################################################################
        # Top 5 Palabras más utilizadas en las preguntas por cada Área (Formato)
        #######################################################################
        
        st.header("Top 5 Tokens por Formato")
    
        # Asume que 'data' ya está cargado, por ejemplo:
        # data = pd.read_excel("Data RFI.xlsx")
        
        df_top_tokens = generar_top_tokens_por_formato(data, top_n=5)
        
        # Mostrar la tabla en Streamlit
        #st.table(df_top_tokens)
        st.dataframe(df_top_tokens, use_container_width=True)

        ###################
        # StopWords
        ###################
       
        st.header("Top 10 palabras por Formato (Sin Stopwords)")
        render_top10_words_by_format(data)

        ####################################
        # Correlación entre Formatos (Áreas)
        ####################################
        
        st.header("Gráfico de Correlación Interactiva entre Formatos (Áreas)")
        plot_correlacion(data)

    except Exception as e:
        st.error("Error al procesar el archivo 'Data RFI.xlsx'.")
        st.error(str(e))
        return

    # Ocultar elementos de Streamlit
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
    return True

