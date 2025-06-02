import os
import pandas as pd
import re
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Data RFI.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "Models", "data_rfi_embeddings.pkl")

def createPage():

    # Cargar y limpiar datos
    def cargar_y_limpiar_datos():
        df = pd.read_excel(DATA_PATH)
        df = df[["Pregunta", "Respuesta"]].dropna()
        df["Pregunta"] = df["Pregunta"].apply(lambda x: re.sub(r"_x000D_\\n|\n", " ", str(x)).strip())
        df["Respuesta"] = df["Respuesta"].apply(lambda x: re.sub(r"_x000D_\\n|\n", " ", str(x)).strip())
        return df

    # Generar embeddings y guardar en .pkl
    def generar_y_guardar_embeddings(df):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(df["Pregunta"].tolist(), show_progress_bar=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({
                "dataframe": df,
                "embeddings": embeddings
            }, f)

    # Cargar modelo solo una vez
    @st.cache_resource
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    # Cargar embeddings y dataframe
    @st.cache_data
    def load_data_embeddings():
        if not os.path.exists(MODEL_PATH):
            df = cargar_y_limpiar_datos()
            generar_y_guardar_embeddings(df)
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        return data["dataframe"], data["embeddings"]

    # Buscar respuesta similar
    def buscar_respuesta(pregunta_usuario, model, df, embeddings):
        pregunta_vec = model.encode([pregunta_usuario])
        similitudes = cosine_similarity(pregunta_vec, embeddings)[0]
        idx_mejor = similitudes.argmax()
        return df.iloc[idx_mejor]["Respuesta"]

    try:
        st.title("🤖 Asistente Virtual DIPRO")
        df, embeddings = load_data_embeddings()
        model = load_model()

        pregunta_usuario = st.text_input("Haz tu pregunta relacionada al proyecto:")
        if pregunta_usuario:
            respuesta = buscar_respuesta(pregunta_usuario, model, df, embeddings)
            st.markdown("### 📌 Respuesta:")
            st.success(respuesta)

    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'Data RFI.xlsx'.")
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
    
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
    