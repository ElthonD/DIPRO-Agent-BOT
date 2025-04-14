# Líbrerias
# ==============================================================================

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

# Para Clustering
# =============================================================================
from sklearn.cluster import AgglomerativeClustering
# Para cálculos de similitud (en caso de necesitar ajustar)
from sklearn.metrics.pairwise import cosine_similarity

# Importar modelo para embeddings de oraciones (compatible con español)
#=============================================================================
from sentence_transformers import SentenceTransformer

# Preprocesado y modelado
# ============================================================================

from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Descargar recursos necesarios de nltk (ejecutar una sola vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Ruta fija del archivo Excel
XLSX_FILE = r'D:\Proyectos\DIPRO Agent BOT\Data\RFI Data.xlsx'

def createPage():

    # Función para cargar datos desde Excel (se usa caché de Streamlit)
    @st.cache_data(show_spinner='Cargando Datos... Espere...', persist=True)
    def load_df():
        df = pd.read_excel(XLSX_FILE)
        df['Fecha/Hora de creación'] = pd.to_datetime(df['Fecha/Hora de creación'])
        
        # Si existe la columna "ID", se renombra a "Registro" y se usa como identificador;
        # de lo contrario, se utiliza el índice.
        if 'ID' in df.columns:
            df.rename(columns={'ID': 'Registro'}, inplace=True)
        else:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Registro'}, inplace=True)
            
        return df
    
    # Función para preprocesar texto: minúsculas, tokenización y eliminación de stopwords y puntuación
    def preprocess_text(text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        tokens = word_tokenize(text, language='spanish')
        stop_words = set(stopwords.words("spanish"))
        # Se añade la stoprword: amp, ax, ex
        stop_words.extend(("amp", "xa", "xe", "Buenos", "días", "tardes", "noches", "Hola", "Gracias","para","la", "de", "el", "en", "a", "y", "que", "es", "se", "por", "respecto", "con", "un", "una", "los", "las", "del", "al", "este", "esta", "esto", "esto", "aquí", "ahí", "allí", "indica", "se", "han","valioso", "apoyo", "atención", "respuesta", "consultas", "duda", "pregunta", "respuesta", "solicitud", "requerimiento", "información", "proyecto", "propuesta", "documento", "anexo", "adjunto", "contratista", "referencia", "alcance", "términos", "condiciones", "contrato", "licitación", "oferta", "evaluación", "proceso", "plazo", "fecha", "hora", "reunión", "presentación", "requerido", "necesario", "sugerencia", "comentario", "observación", "darnos", "respuesta", "favor", "enviar", "adjuntar", "comunicación", "correo", "electrónico", "respuesta", "solicitud", "requerimiento", "información", "proyecto", "propuesta", "documento", "anexo", "adjunto", "contratista", "referencia", "alcance", "términos", "condiciones", "contrato", "licitación", "oferta", "evaluación", "proceso", "referente"))
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        return " ".join(tokens)

    # Limpieza de columnas importantes: se agregan las versiones limpias de preguntas y respuestas
    def limpieza_columnas_importantes(df):
        df['Pregunta_clean'] = df['Pregunta del RFI'].apply(preprocess_text)
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
        
        # Realizar clustering para las preguntas
        clustering_questions = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.7,   # Valor ajustable
            metric='cosine',
            linkage='average'
        )
        df['Pregunta_Group_ID'] = clustering_questions.fit_predict(question_embeddings)
        
        # Realizar clustering para las respuestas
        clustering_answers = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.7,   # Valor ajustable
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
        st.dataframe(freq_preg)
        
        top_80 = freq_preg[freq_preg['Cumulative_Percent'] <= 0.8]
        st.markdown("<h3 style='text-align: left;'>Grupos que representan el 80% de las consultas:</h3>", unsafe_allow_html=True)
        st.dataframe(top_80)
    
    try:
        st.markdown("<h3 style='text-align: left;'>Data</h3>", unsafe_allow_html=True)
        data = load_df()
        st.dataframe(data)

        st.markdown("<h3 style='text-align: left;'>Análisis de la Data</h3>", unsafe_allow_html=True)
        st.write(""" 
        Pasos a seguir para esta sección:
        1. Aplicar la limpieza a las columnas relevantes.
        2. Generar representaciones vectoriales que permitan comparar semánticamente las oraciones de las preguntas y respuestas.
        3. Contar la cantidad de registros que comparten el mismo grupo (tanto para preguntas como para respuestas) y se añade esa información a cada registro.
        4. Se crea un DataFrame final que contiene las siguientes columnas:
            + Registro: Número de registro original.
            + Pregunta del RFI: Pregunta original.
            + Pregunta_Group_ID: Identificador del grupo semántico de la pregunta.
            + Respuesta: Respuesta original.
            + Respuesta_Group_ID: Identificador del grupo semántico de la respuesta.
            + Pregunta_Frequency: Número de veces que se repite esa misma duda (grupo).
            + Respuesta_Frequency: Número de veces que se repite esa respuesta (grupo).            
        5. Análisis exploratorio: 80/20 y diagnóstico inicial.       
        """)
        
        # Aplicar limpieza a las columnas importantes
        data_limpia = limpieza_columnas_importantes(data)
        
        # Calcular el hash del archivo para detectar si existen nuevos datos
        file_hash = get_file_hash(XLSX_FILE)
        
        # Entrenar (generar embeddings y clustering) solo si los datos han cambiado
        data_cluster = compute_embeddings_and_clustering(data_limpia, file_hash)
        
        # Calcular frecuencias por grupo y reestructurar el DataFrame final
        data_final = calculo_frecuencia_grupo(data_cluster)
        final_df = reestructurar_dataframe(data_final)
        
        # Mostrar el DataFrame final en Streamlit
        st.dataframe(final_df)
        
        # Convertir a Excel (en memoria) y proporcionar botón para descargarlo
        excel_data = convertir_a_excel(final_df)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.download_button(
                label="Descargar Excel",
                data=excel_data,
                file_name="RFI_Data_Analizado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Análisis Pareto de la data
        analisis_pareto(final_df)

    except Exception as e:
        st.error("Error al cargar el archivo 'RFI Data.xlsx'. Asegúrate de que se encuentre en el directorio correcto.")
        st.error(str(e))
        return
    
    except UnboundLocalError as e:
        print("Seleccionar: ", e)

    except ZeroDivisionError as e:
        print("Seleccionar: ", e)
    
    except KeyError as e:
        print("Seleccionar: ", e)

    except ValueError as e:
        print("Seleccionar: ", e)
    
    except IndexError as e:
        print("Seleccionar: ", e)

     # ---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    return True 