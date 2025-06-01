import os
import streamlit as st
import pandas as pd
from llama_cpp import Llama
import joblib
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Data RFI.xlsx")
LLM_MODEL_PATH = os.path.join(BASE_DIR, "Models", "llama-2-7b-chat.Q2_K.gguf")
SEMANTIC_API_URL = "http://localhost:8000/search"  # Cambia el puerto si usas otro

def createPage():
    @st.cache_data
    def load_data():
        df = pd.read_excel(DATA_PATH)
        return df.astype(str)

    @st.cache_resource
    def get_llm():
        return Llama(model_path=LLM_MODEL_PATH, n_ctx=128)

    try:
        st.title("🤖 Asistente Virtual DIPRO")
        df = load_data()
        if 'Respuesta' in df.columns:
            texts = df['Respuesta'].tolist()
        else:
            st.error("No se encontró la columna 'Respuesta' en el archivo Excel.")
            return

        llm = get_llm()
        st.subheader("Asistente Virtual")
        user_query = st.text_input("Haz tu pregunta:")

        if user_query:
            # Llama al microservicio para obtener los índices relevantes
            response = requests.post(
                SEMANTIC_API_URL,
                json={"question": user_query, "top_k": 5}
            )
            if response.status_code == 200:
                indices = response.json()["indices"]
                context_respuestas = "\n".join([texts[i] for i in indices])
            else:
                context_respuestas = "No se pudo obtener contexto del microservicio."

            prompt = (
                "Eres un asistente experto en temas de la empresa DIPRO. "
                "A continuación tienes información relevante extraída de la base de datos de respuestas oficiales de la empresa. "
                "Utiliza este contexto para responder la pregunta del usuario de manera clara, profesional y natural. "
                "No repitas literalmente el texto del contexto, sino que explica, resume o adapta la información para que sea útil y fácil de entender. "
                "Si hay varias respuestas relevantes, integra la información de forma coherente. "
                "Si el contexto no es suficiente, responde lo mejor posible según tu conocimiento general, pero prioriza siempre la información proporcionada.\n\n"
                f"Contexto:\n{context_respuestas}\n\n"
                f"Pregunta del usuario: {user_query}\n"
                "Respuesta:"
            )

            response_llm = llm(prompt=prompt, max_tokens=256, temperature=0.85)
            st.markdown(f"**Asistente:** {response_llm['choices'][0]['text'].strip()}")

    except Exception as e:
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