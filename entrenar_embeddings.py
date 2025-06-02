# Requisitos: pip install pandas sentence-transformers openpyxl
import os
import pandas as pd
import re
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Data RFI.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "Models", "data_rfi_embeddings.pkl")

# Paso 1: Cargar y limpiar datos
def cargar_y_limpiar_datos():
    df = pd.read_excel(DATA_PATH)
    df = df[["Pregunta", "Respuesta"]].dropna()
    df["Pregunta"] = df["Pregunta"].apply(lambda x: re.sub(r"_x000D_\\n|\n", " ", str(x)).strip())
    df["Respuesta"] = df["Respuesta"].apply(lambda x: re.sub(r"_x000D_\\n|\n", " ", str(x)).strip())
    return df

# Paso 2: Generar embeddings y guardar en .pkl
def generar_y_guardar_embeddings(df):
    print("Cargando modelo de embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Generando embeddings...")
    embeddings = model.encode(df["Pregunta"].tolist(), show_progress_bar=True)
    
    print(f"Guardando datos y embeddings en {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "dataframe": df,
            "embeddings": embeddings
        }, f)
    print("✅ Listo. Archivo guardado.")

# Ejecutar proceso completo
if __name__ == "__main__":
    df = cargar_y_limpiar_datos()
    generar_y_guardar_embeddings(df)
