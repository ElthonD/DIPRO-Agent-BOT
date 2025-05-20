import os
import string
import hashlib
import pickle

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import faiss

# ─── Configuración de rutas ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "Data", "Data RFI.xlsx")
LOGO_PATH   = os.path.join(BASE_DIR, "Imagenes", "Dipro_Logo1.png")
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
EMB_CACHE   = os.path.join(CACHE_DIR, "embeddings_cache.pkl")
IDX_CACHE   = os.path.join(CACHE_DIR, "faiss.index")
DF_CACHE    = os.path.join(CACHE_DIR, "data_rfi.pkl")

def createPage():

    # ─── Asegurar carpeta de cache ────────────────────────────────────────────────
    def ensure_cache_dir():
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ─── NLP Preprocesado ─────────────────────────────────────────────────────────
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    _spanish_stop = set(stopwords.words("spanish"))

    def preprocess_text(text: str) -> str:
        text = text.lower()
        tokens = word_tokenize(text, language='spanish')
        tokens = [t for t in tokens if t not in _spanish_stop and t not in string.punctuation]
        return " ".join(tokens)

    # ─── Hash del archivo para invalidar cache ────────────────────────────────────
    def file_md5(path: str) -> str:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    # ─── Carga / cache del DataFrame ─────────────────────────────────────────────
    def load_or_cache_df() -> pd.DataFrame:
        ensure_cache_dir()
        if os.path.exists(DF_CACHE):
            df = pd.read_pickle(DF_CACHE)
        else:
            df = pd.read_excel(DATA_PATH).dropna(subset=['Pregunta','Respuesta'])
            df['Pregunta_proc'] = df['Pregunta'].map(preprocess_text)
            df.to_pickle(DF_CACHE)
        return df

    # ─── Modelo de embeddings ────────────────────────────────────────────────────
    @st.cache_resource(show_spinner=False)
    def get_embedding_model() -> SentenceTransformer:
        return SentenceTransformer("all-mpnet-base-v2")  # más preciso que miniLM

    # ─── Carga / cálculo de embeddings con cache MD5 ─────────────────────────────
    def load_or_compute_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
        ensure_cache_dir()
        current_hash = file_md5(DATA_PATH)
        if os.path.exists(EMB_CACHE):
            cache = pickle.load(open(EMB_CACHE, 'rb'))
            if cache.get('hash') == current_hash:
                st.info("🔄 Cargando embeddings desde cache")
                return cache['embeddings']
        # si no hay cache o cambió el archivo:
        st.info("⚙️ Generando nuevos embeddings por cambio en Data RFI")
        embeddings = model.encode(df['Pregunta_proc'].tolist(), convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        pickle.dump({'hash': current_hash, 'embeddings': embeddings}, open(EMB_CACHE, 'wb'))
        return embeddings

    # ─── Carga / construcción de índice FAISS ─────────────────────────────────────
    def load_or_build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
        ensure_cache_dir()
        if os.path.exists(IDX_CACHE):
            index = faiss.read_index(IDX_CACHE)
        else:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            faiss.write_index(index, IDX_CACHE)
        return index

    # ─── Recuperación de respuesta más similar ───────────────────────────────────
    def retrieve_answer(query: str,
                        df: pd.DataFrame,
                        model: SentenceTransformer,
                        index: faiss.IndexFlatL2) -> str:
        q_proc = preprocess_text(query)
        q_emb  = model.encode([q_proc], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        D, I   = index.search(q_emb, 1)
        idx    = int(I[0][0])
        score  = 1 - D[0][0]  # similitud aproximada
        if score < 0.5:
            return "Lo siento, no tengo una respuesta precisa para esa pregunta."
        return df.iloc[idx]['Respuesta']

        # Logo
        img = Image.open(LOGO_PATH).convert("RGBA")
        col1, col2, col3 = st.columns([1.5,1,1])
        with col2:
            st.image(img, width=150)

        st.title("🤖 ChatBot DIPRO")
        st.write("Escribe tu pregunta en el campo y recibirás la respuesta histórica más relevante.")

        # Carga datos + modelo + embeddings + índice
        df        = load_or_cache_df()
        model     = get_embedding_model()
        embeddings= load_or_compute_embeddings(df, model)
        index     = load_or_build_index(embeddings)
        st.success(f"Modelo listo con {len(df)} pares Pregunta/Respuesta")

        # Chat
        if 'history' not in st.session_state:
            st.session_state.history = []

        pregunta = st.text_input("Tu pregunta:")
        if st.button("Enviar"):
            if pregunta.strip():
                respuesta = retrieve_answer(pregunta, df, model, index)
                st.session_state.history.append((pregunta, respuesta))
            else:
                st.warning("Escribe algo primero…")

        for q, a in st.session_state.history:
            st.markdown(f"**Tú:** {q}")
            st.markdown(f"**BOT DIPRO:** {a}")
            st.markdown("---")

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

