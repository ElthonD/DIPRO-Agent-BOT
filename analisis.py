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
        
        # Calcular la frecuencia de cada valor en la columna 'Área'
        conteo = df['Formato'].value_counts().reset_index()
        conteo.columns = ['Formato', 'Frecuencia']

        # Ordenar de mayor a menor frecuencia
        conteo = conteo.sort_values(by='Frecuencia', ascending=False)

        # Calcular la suma acumulada y el porcentaje acumulado
        conteo['Acumulado'] = conteo['Frecuencia'].cumsum()
        total = conteo['Frecuencia'].sum()
        conteo['Porcentaje Acumulado'] = 100 * conteo['Acumulado'] / total

        # Crear la figura de Plotly
        fig = go.Figure()

        # Agregar el gráfico de barras para la frecuencia
        fig.add_trace(go.Bar(
            x=conteo['Formato'],
            y=conteo['Frecuencia'],
            name='Frecuencia',
            text=conteo["Frecuencia"],
            textposition="outside",
        ))

        # Agregar el gráfico de línea para el porcentaje acumulado
        fig.add_trace(go.Scatter(
            x=conteo['Formato'],
            y=conteo['Porcentaje Acumulado'],
            name='Porcentaje Acumulado',
            mode='lines+markers',
            text=conteo["Frecuencia"],
            textposition="top center",
            yaxis='y2'
        ))

        # Configurar el layout para tener dos ejes Y
        fig.update_layout(
            title='Diagrama de Pareto',
            xaxis=dict(title='Formato'),
            yaxis=dict(title='Frecuencia'),
            yaxis2=dict(
                title='Porcentaje Acumulado (%)',
                overlaying='y',
                side='right',
                range=[0, 110]
            ),
            legend=dict(x=0.75, y=1.15)
        )
        
        # Mostrar la gráfica
        fig.show()

   
        
    ############################################
    # Funciones Auxiliares para el Preprocesado
    ############################################

    def extract_common_stopwords(df, col1="Pregunta", col2="Respuesta", top_n=10):
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
    common_stopwords = extract_common_stopwords(data, col1="Pregunta", col2="Respuesta", top_n=10)
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
        stop_words.update(("amp", "xa", "xe", "plano", "indica", "proyecto", "si", "apoyo", "área", "favor", "acuerdo", "detalle", "solicita", "rfi", "area", "si", "existente", "buena", "oc", "cm", "aunado", "indicar", "referente", "trabajos", "tarde", "solicito", "cambio", "hallazgo", "adjunta", "producto", "nuevo", "solicitamos", "indiquen", "ser", "confirmar", "embargo", "procede", "ie", "indicarnos", "realizar", "confirmar", "procede"))
        #stop_words.update(("amp", "xa", "xe", "Buenos", "días", "tardes", "noches", ...))
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        return " ".join(tokens)

    # Limpieza de columnas importantes: se agregan las versiones limpias de preguntas y respuestas
    def limpieza_columnas_importantes(df):
        df['Pregunta_clean'] = df['Pregunta'].apply(preprocess_text)
        # Asegurarse de que la columna de respuestas se llame correctamente (puede ser "Respuestas" o "Respuesta")
        df['Respuesta_clean'] = df['Respuesta'].apply(preprocess_text)
        return df

    # Función que calcula el hash MD5 del archivo para detectar cambios
    def get_file_hash(file_path):
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        return hashlib.md5(file_bytes).hexdigest()

    def load_or_train_model(df):
        """
        Esta función verifica si existe un modelo persistente (almacenado en un pickle)
        y si el hash del archivo de datos coincide con el almacenado. De ser así, carga el resultado;
        de lo contrario, entrena el modelo (genera embeddings y ejecuta clustering) y lo guarda.
        """
        file_hash = get_file_hash(DATA_PATH)
        
        # Verificar si existe un modelo almacenado
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    modelo_data = pickle.load(f)
                # Verificar que el hash almacenado coincide
                if modelo_data.get("file_hash") == file_hash:
                    # Se ha precargado el modelo: no es necesario reentrenar
                    return modelo_data["df_clustered"], file_hash
            except Exception as e:
                # Si falla la carga, se procede a reentrenar
                print("Error al cargar el modelo persistente:", e)
        
        # Si se llega aquí es porque no existe modelo o se detectaron cambios: se entrena de nuevo
        sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Generar embeddings para la columna limpia de preguntas y respuestas
        question_embeddings = sentence_model.encode(df['Pregunta_clean'].tolist())
        answer_embeddings = sentence_model.encode(df['Respuesta_clean'].tolist())
        
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
        
        # Se pueden incluir aquí pasos adicionales (p.ej., cálculo de frecuencias)
        
        # Guardar el resultado junto con el hash de datos en un diccionario
        modelo_data = {"file_hash": file_hash, "df_clustered": df}
        try:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(modelo_data, f)
        except Exception as e:
            print("Error al guardar el modelo persistente:", e)
        
        return df, file_hash

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
        group_preg_freq = df.groupby('Pregunta_Group_ID')['Pregunta'].count().rename('Pregunta_Frequency')
        group_resp_freq = df.groupby('Respuesta_Group_ID')['Respuesta'].count().rename('Respuesta_Frequency')
        df = df.merge(group_preg_freq, on='Pregunta_Group_ID', how='left')
        df = df.merge(group_resp_freq, on='Respuesta_Group_ID', how='left')
        return df

    # Función para reestructurar el DataFrame final con las columnas requeridas
    def reestructurar_dataframe(df):
        final_df = df[['Registro', 'Pregunta', 'Pregunta_Group_ID', 
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
        
        # Carga de Data
        st.markdown("<h3 style='text-align: left;'>Data</h3>", unsafe_allow_html=True)
        st.dataframe(data)
        
        # Cuantificar Áreas (Formato)
        st.markdown("<h3 style='text-align: left;'>Cuantificar Áreas (Formato)</h3>", unsafe_allow_html=True)
        cantidad_areas = (len(data['Formato'].unique()) - 1)
        lista_areas = data['Formato'].unique().tolist()
        lista = lista_areas.pop(-1)
        st.write(f"La cantidad de áreas (Formato) presentes en el documento son:" + str(cantidad_areas) + ". Las áreas (Formato) presentes son: " + str(lista_areas) + ".")
        
        # Diagrama de Pareto
        st.markdown("<h3 style='text-align: left;'>Gráfico Pareto</h3>", unsafe_allow_html=True)
        pareto_areas1 = diagrama_pareto(data)

        """
        # Aplicar limpieza a las columnas importantes
        data_limpia = limpieza_columnas_importantes(data)

        # Cargar o entrenar el modelo persistente (basado en el hash del archivo)
        data_clustered, file_hash = load_or_train_model(data_limpia)
        
        # Entrenar (generar embeddings y clustering) solo si hay cambios
        #data_cluster = compute_embeddings_and_clustering(data_limpia, file_hash)
        
        # Calcular frecuencias por grupo y reestructurar el DataFrame final
        data_final = calculo_frecuencia_grupo(data_clustered)
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
        """
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

