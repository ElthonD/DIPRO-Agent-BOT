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
import xlsxwriter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas relativas portables
DATA_PATH = os.path.join(BASE_DIR, "Data", "RFI Data.xlsx")
LOGO_PATH = os.path.join(BASE_DIR, "Imagenes", "Dipro_Logo1.png")

############################################
# Función Principal de la Página
############################################

def createPage():
    
    # Función para cargar datos desde Excel (usando caché de Streamlit)
    @st.cache_data(show_spinner='Cargando Datos... Espere...', persist=True)
    def load_df():
        df = pd.read_excel(DATA_PATH)
        df['Fecha/Hora de creación'] = pd.to_datetime(df['Fecha/Hora de creación'])
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
        st.error("Error al cargar el archivo 'RFI Data.xlsx'.")
        st.error(str(e))
        return

    st.markdown("<h3 style='text-align: left;'>Data</h3>", unsafe_allow_html=True)
    st.dataframe(data)
    
    ############################################
    # Funciones Auxiliares para el Preprocesado
    ############################################

    def extract_common_stopwords(df, col1="Pregunta del RFI", col2="Respuesta", top_n=10):
        """
        Extrae de las columnas especificadas los tokens que ya se
        encuentran en la lista de stopwords de nltk y retorna una lista
        de las 'top_n' stopwords más comunes.
        """
        # Concatenar el contenido de ambas columnas (omitiendo valores nulos)
        text = " ".join(df[col1].dropna().astype(str).tolist() + df[col2].dropna().astype(str).tolist()).lower()
        tokens = word_tokenize(text, language="spanish")
        # Filtrar para excluir puntuación
        tokens = [token for token in tokens if token not in string.punctuation]
        # Obtener la lista por defecto de stopwords en español
        default_stopwords = set(stopwords.words("spanish"))
        # Seleccionar solo los tokens que ya sean stopwords
        stopword_tokens = [token for token in tokens if token in default_stopwords]
        
        from collections import Counter
        counter = Counter(stopword_tokens)
        common_stop = [word for word, count in counter.most_common(top_n)]
        return common_stop

    # Extraer automáticamente las stopwords más comunes de las columnas de interés
    common_stopwords = extract_common_stopwords(data, col1="Pregunta del RFI", col2="Respuesta", top_n=10)
    #st.write("Stopwords más comunes extraídas:", common_stopwords)
    
    # Función para preprocesar texto: minúsculas, tokenización y eliminación de stopwords y puntuación
    def preprocess_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        tokens = word_tokenize(text, language='spanish')
        # Obtener el conjunto de stopwords por defecto
        stop_words = set(stopwords.words("spanish"))
        # Agregar las stopwords más comunes extraídas del archivo
        stop_words.update(common_stopwords)
        # También se pueden agregar palabras adicionales manualmente
        stop_words.update(("amp", "xa", "xe", "Buenos", "días", "tardes", "noches", "Hola", "Gracias","para","la", "de", "el", "en", "a", "y", "que", "es", "se", "por", "respecto", "con", "un", "una", "los", "las", "del", "al", "este", "esta", "esto", "esto", "aquí", "ahí", "allí", "indica", "se", "han","valioso", "apoyo", "atención", "respuesta", "consultas", "duda", "pregunta", "respuesta", "solicitud", "requerimiento", "información", "proyecto", "propuesta", "documento", "anexo", "adjunto", "contratista", "referencia", "alcance", "términos", "condiciones", "contrato", "licitación", "oferta", "evaluación", "proceso", "plazo", "fecha", "hora", "reunión", "presentación", "requerido", "necesario", "sugerencia", "comentario", "observación", "darnos", "respuesta", "favor", "enviar", "adjuntar", "comunicación", "correo", "electrónico", "respuesta", "solicitud", "requerimiento", "información", "proyecto", "propuesta", "documento", "anexo", "adjunto", "contratista", "referencia", "alcance", "términos", "condiciones", "contrato", "licitación", "oferta", "evaluación", "proceso", "referente"))
        #stop_words.update(("amp", "xa", "xe", "Buenos", "días", "tardes", "noches", ...))
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        return " ".join(tokens)

    # Limpieza de columnas importantes: se agregan las versiones limpias de preguntas y respuestas
    def limpieza_columnas_importantes(df):
        df['Pregunta_clean'] = df['Pregunta del RFI'].apply(preprocess_text)
        # Asegurarse de que la columna de respuestas se llame correctamente (puede ser "Respuestas" o "Respuesta")
        df['Respuesta_clean'] = df['Respuesta'].apply(preprocess_text)
        return df

    # Función que calcula el hash MD5 del archivo para detectar cambios
    def get_file_hash(file_path):
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        return hashlib.md5(file_bytes).hexdigest()

    # Función que genera embeddings y realiza clustering solo cuando hay nuevos datos
    @st.cache_data(show_spinner="Entrenando modelo...", persist=True)
    def compute_embeddings_and_clustering(df, file_hash):
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # Generar embeddings para las columnas limpias
        question_embeddings = model.encode(df['Pregunta_clean'].tolist())
        answer_embeddings = model.encode(df['Respuesta_clean'].tolist())
        # Clustering para preguntas
        clustering_questions = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.7,
            metric='cosine',
            linkage='average'
        )
        df['Pregunta_Group_ID'] = clustering_questions.fit_predict(question_embeddings)
        # Clustering para respuestas
        clustering_answers = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.7,
            metric='cosine',
            linkage='average'
        )
        df['Respuesta_Group_ID'] = clustering_answers.fit_predict(answer_embeddings)
        return df

    # Función para calcular la frecuencia de cada grupo y fusionarla al DataFrame
    def calculo_frecuencia_grupo(df):
        group_preg_freq = df.groupby('Pregunta_Group_ID')['Pregunta del RFI'].count().rename('Pregunta_Frequency')
        group_resp_freq = df.groupby('Respuesta_Group_ID')['Respuesta'].count().rename('Respuesta_Frequency')
        df = df.merge(group_preg_freq, on='Pregunta_Group_ID', how='left')
        df = df.merge(group_resp_freq, on='Respuesta_Group_ID', how='left')
        return df

    # Función para reestructurar el DataFrame final con las columnas requeridas
    def reestructurar_dataframe(df):
        final_df = df[['Registro', 'Pregunta del RFI', 'Pregunta_Group_ID', 
                       'Respuesta', 'Respuesta_Group_ID', 'Pregunta_Frequency', 'Respuesta_Frequency']]
        return final_df

    # Función para convertir el DataFrame a Excel en memoria
    def convertir_a_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Datos')
            writer.close()
        processed_data = output.getvalue()
        return processed_data

    # Función para realizar un análisis Pareto y mostrar resultados en Streamlit
    def analisis_pareto(df):
        n_dudas = df['Pregunta_Group_ID'].nunique()
        n_respuestas = df['Respuesta_Group_ID'].nunique()
        st.write(f"Número total de dudas distintas (grupos de preguntas): {n_dudas}")
        st.write(f"Número total de grupos de respuestas distintas: {n_respuestas}")
        freq_preg = df.groupby('Pregunta_Group_ID').size().reset_index(name='Count')
        freq_preg = freq_preg.sort_values(by='Count', ascending=False)
        freq_preg['Cumsum'] = freq_preg['Count'].cumsum()
        freq_preg['Cumulative_Percent'] = freq_preg['Cumsum'] / freq_preg['Count'].sum()
        st.markdown("<h3 style='text-align: left;'>Distribución de las frecuencias de las dudas</h3>", unsafe_allow_html=True)
        st.dataframe(freq_preg, hide_index=True)
        top_80 = freq_preg[freq_preg['Cumulative_Percent'] <= 0.8]
        #st.markdown("<h3 style='text-align: left;'>Grupos que representan el 80% de las consultas:</h3>", unsafe_allow_html=True)
        #st.dataframe(top_80)
        return freq_preg

    def plot_pareto_chart(freq_preg):
   
        # Crear la figura con eje secundario para el acumulado
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Agregar la gráfica de barras para la frecuencia
        fig.add_trace(
            go.Bar(
                x=freq_preg['Pregunta_Group_ID'],
                y=freq_preg['Count'],
                name="Frecuencia",
                hovertemplate="Grupo: %{x}<br>Frecuencia: %{y}<extra></extra>"
            ),
            secondary_y=False,
        )
        
        # Agregar la línea para el porcentaje acumulado
        fig.add_trace(
            go.Scatter(
                x=freq_preg['Pregunta_Group_ID'],
                y=freq_preg['Cumulative_Percent'] * 100,
                name="Acumulado (%)",
                mode="lines+markers",
                hovertemplate="Grupo: %{x}<br>Acumulado: %{y:.2f}%<extra></extra>"
            ),
            secondary_y=True,
        )
        
        # Actualizar el layout y forzar el orden del eje x según el array de los group_id ordenados por frecuencia
        fig.update_layout(
            title="Gráfica Pareto de las dudas (Preguntas)",
            xaxis_title="Grupo (ordenado de mayor a menor frecuencia)",
            yaxis_title="Frecuencia",
            template="plotly_white",
            legend=dict(x=0.75, y=1.15),
            xaxis=dict(
                categoryorder="array",
                categoryarray=freq_preg['Pregunta_Group_ID'].tolist()  # Orden personalizado
            )
        )
        
        # Configurar el eje secundario para el porcentaje acumulado
        fig.update_yaxes(title_text="Porcentaje Acumulado (%)", secondary_y=True)
        
        return fig


    try:
        st.markdown("<h3 style='text-align: left;'>Análisis de la Data</h3>", unsafe_allow_html=True)
        st.write(""" 
        Pasos a seguir:
        1. Limpiar las columnas relevantes.
        2. Generar embeddings y realizar clustering (solo si hay nuevos datos).
        3. Calcular la frecuencia por grupo y reestructurar el DataFrame final.
        4. Exportar el resultado a Excel y realizar un análisis exploratorio (Pareto).
        """)
        
        # Aplicar limpieza a las columnas importantes
        data_limpia = limpieza_columnas_importantes(data)
        # Calcular el hash del archivo para detectar cambios en los datos
        file_hash = get_file_hash(DATA_PATH)
        # Entrenar (generar embeddings y clustering) solo si hay cambios
        data_cluster = compute_embeddings_and_clustering(data_limpia, file_hash)
        # Calcular frecuencias por grupo y reestructurar el DataFrame final
        data_final = calculo_frecuencia_grupo(data_cluster)
        final_df = reestructurar_dataframe(data_final)
        
        # Mostrar el DataFrame final
        st.dataframe(final_df, hide_index=True)
        
        # Convertir a Excel en memoria y proporcionar botón de descarga (botón centrado)
        excel_data = convertir_a_excel(final_df)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.download_button(
                label="Descargar Excel",
                data=excel_data,
                file_name="RFI_Data_Analizado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        # Análisis Pareto
        #analisis_pareto(final_df)

        # Ejecutar la función analisis_pareto para obtener la distribución
        freq_preg = analisis_pareto(final_df)

        # Generar la gráfica Pareto con Plotly
        #fig = plot_pareto_chart(freq_preg)

        # Mostrar la gráfica en Streamlit
        #st.markdown("<h3 style='text-align: left;'>Grupos que representan el 80% de las consultas:</h3>", unsafe_allow_html=True)
        #st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("Error al procesar el archivo 'RFI Data.xlsx'.")
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

