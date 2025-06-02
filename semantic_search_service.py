import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "Models", "embeddings.joblib")
NN_MODEL_FILE = os.path.join(BASE_DIR, "Models", "nn_model.joblib")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

try:
    print("Cargando embeddings y modelo...")
    embeddings = joblib.load(EMBEDDINGS_FILE)
    nn = joblib.load(NN_MODEL_FILE)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    print("¡Listo!")
except Exception as e:
    print(f"Error al cargar modelos: {e}")
    raise RuntimeError("No se pudieron cargar los modelos o embeddings.")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/search")
def search(request: QueryRequest):
    try:
        q_emb = embed_model.encode([request.question]).astype('float32')
        distances, indices = nn.kneighbors(q_emb, n_neighbors=request.top_k)
        return {"indices": indices[0].tolist(), "distances": distances[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))