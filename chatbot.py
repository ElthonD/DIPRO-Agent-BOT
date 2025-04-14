import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image
import string
import hashlib
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util


# Descargar recursos necesarios de nltk (ejecutar una sola vez)
nltk.download('punkt')
nltk.download('stopwords')

os.environ["STREAMLIT_WATCHER_PATCH"] = "true"
# Ruta base del proyecto (misma carpeta donde está este archivo .py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas relativas portables
DATA_PATH = os.path.join(BASE_DIR, "Data", "RFI Data.xlsx")
LOGO_PATH = os.path.join(BASE_DIR, "Imagenes", "Dipro_Logo1.png")
CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache.pkl")


def createPage():

    # Title of the main page
    #pathLogo = pathLogo = r'D:\Proyectos\DIPRO Agent BOT\Imagenes\Dipro_Logo1.png'
    # Abrir imagen y convertirla a RGB (3 canales)
     # Abrir imagen con canal alfa
    img = Image.open(LOGO_PATH).convert("RGBA")
    # Convertir a numpy array
    #display = np.array(display)
    col1, col2, col3 = st.columns([1.5,1,1])
    with col2:
        st.image(img, use_container_width=False)

    #############################
    # Funciones de Preprocesamiento
    #############################

    def preprocess_text(text):
        """
        Convierte el texto a minúsculas, tokeniza y elimina stopwords y signos de puntuación.
        """
        text = text.lower()
        tokens = word_tokenize(text, language='spanish')
        stop_words = set(stopwords.words("spanish"))
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        return " ".join(tokens)
    
    #############################
    # Funciones para Cargar Datos
    #############################

    @st.cache_data(show_spinner=False)
    def load_data():
        """
        Carga los datos del Excel, extrae las columnas y preprocesa las preguntas.
        """
        #xlsx_data = r'D:\Proyectos\DIPRO Agent BOT\Data\RFI Data.xlsx'
        df = pd.read_excel(DATA_PATH)
        preguntas_raw = df["Pregunta del RFI"].tolist()
        respuestas = df["Respuesta"].tolist()
        preguntas_preprocesadas = [preprocess_text(p) for p in preguntas_raw]
        return preguntas_raw, respuestas, preguntas_preprocesadas

    #############################
    # Funciones para el Modelo y Embeddings
    #############################

    @st.cache_resource(show_spinner=False)
    def load_model():
        """
        Carga el modelo de SentenceTransformer.
        Se utiliza "all-MiniLM-L6-v2", un modelo compacto y eficiente para tareas de similitud semántica.
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model

    def get_file_hash(file_path):
        """
        Calcula y retorna el hash MD5 del archivo en modo binario.
        """
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        return hashlib.md5(file_bytes).hexdigest()
    
    def load_embeddings_cache(cache_path="embeddings_cache.pkl"):
        """
        Carga la cache de embeddings si existe.
        """
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            return cache
        return None

    def save_embeddings_cache(embeddings, file_hash, cache_path="embeddings_cache.pkl"):
        """
        Guarda en cache los embeddings junto con el hash del archivo.
        """
        cache = {
        "file_hash": file_hash,
        "embeddings": embeddings
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    def get_embeddings_with_cache(model, preguntas_preprocesadas, data_file="RFI Data.xlsx", cache_path="embeddings_cache.pkl"):
        """
        Genera los embeddings de las preguntas preprocesadas utilizando una cache persistente:
        - Si el hash del archivo coincide con el de la cache, se cargan los embeddings precomputados.
        - Si hay cambios en el archivo, se generan nuevos embeddings y se actualiza la cache.
        """
        current_hash = get_file_hash(data_file)
        cache = load_embeddings_cache(cache_path)
        
        if cache is not None and cache.get("file_hash") == current_hash:
            embeddings = cache.get("embeddings")
            st.info("Cargando embeddings desde la cache.")
        else:
            st.info("Generando nuevos embeddings por cambios en los datos.")
            embeddings = model.encode(preguntas_preprocesadas, convert_to_tensor=True)
            save_embeddings_cache(embeddings, current_hash, cache_path)
        return embeddings

    @st.cache_data(show_spinner=False)
    def get_embeddings(_model, preguntas_preprocesadas, model_name: str):
        """
        Genera los embeddings de las preguntas preprocesadas utilizando el modelo.
        Usa `model_name` para que Streamlit pueda invalidar el cache si se cambia de modelo.
        """
        embeddings = _model.encode(preguntas_preprocesadas, convert_to_tensor=True)
        return embeddings

    #############################
    # Función para Obtener Respuesta
    #############################

    def obtener_respuesta(pregunta_usuario, modelo, embeddings_preguntas, respuestas, umbral=0.6):
        """
        Preprocesa la pregunta del usuario, genera su embedding y calcula la similitud
        con cada una de las preguntas históricas para recuperar la respuesta asociada.
        """
        pregunta_proc = preprocess_text(pregunta_usuario)
        embedding_usuario = modelo.encode(pregunta_proc, convert_to_tensor=True)
        similitudes = util.cos_sim(embedding_usuario, embeddings_preguntas)[0]
        indice_mejor = int(np.argmax(similitudes))
        puntaje_similitud = similitudes[indice_mejor].item()
        
        if puntaje_similitud < umbral:
            return "Lo siento, no tengo una respuesta precisa para esa pregunta."
        return respuestas[indice_mejor]
   

   #############################
    # ChatBot
    #############################

    st.title("ChatBot DIPRO")
    st.write("Bienvenido al ***Chatbot de DIPRO***. Ingrese una pregunta o escriba 'salir' para terminar la sesión.")

     # Cargar datos (preguntas y respuestas)
    preguntas_raw, respuestas, preguntas_preprocesadas = load_data()
    
    # Cargar el modelo de SentenceTransformer
    modelo = load_model()
    
    # Obtener los embeddings utilizando la cache si es posible
    embeddings_preguntas = get_embeddings_with_cache(modelo, preguntas_preprocesadas, data_file=DATA_PATH, cache_path=CACHE_PATH)

    st.success("Datos, modelo y embeddings cargados correctamente.")
    
    # Campo de entrada para la pregunta del usuario (la respuesta se actualiza automáticamente)
    pregunta_usuario = st.text_input("Escribe tu pregunta:")
    
    if pregunta_usuario:
        respuesta = obtener_respuesta(pregunta_usuario, modelo, embeddings_preguntas, respuestas)
        st.markdown("**Respuesta:**")
        st.write(respuesta)
        
    else:st.warning("Por favor, escribe una pregunta.")

    return True
