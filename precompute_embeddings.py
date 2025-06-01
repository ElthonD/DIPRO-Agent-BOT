import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "Data RFI.xlsx")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "Models", "embeddings.joblib")
NN_MODEL_FILE = os.path.join(BASE_DIR, "Models", "nn_model.joblib")

print("Cargando datos...")
df = pd.read_excel(DATA_PATH).astype(str)
if 'Respuesta' not in df.columns:
    raise ValueError("No se encontró la columna 'Respuesta' en el archivo Excel.")

texts = df['Respuesta'].tolist()

print("Cargando modelo de embeddings...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

print("Calculando embeddings...")
embeddings = embed_model.encode(texts, show_progress_bar=True).astype('float32')

print("Entrenando NearestNeighbors...")
nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(embeddings)

print("Guardando embeddings y modelo...")
joblib.dump(embeddings, EMBEDDINGS_FILE)
joblib.dump(nn, NN_MODEL_FILE)

print("¡Listo! Los archivos embeddings.joblib y nn_model.joblib han sido guardados.")