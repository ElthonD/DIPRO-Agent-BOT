#!/bin/bash
set -e
mkdir -p Models
# Descarga con autenticación usando el token de Hugging Face
wget --header="Authorization: Bearer $HF_TOKEN" -O Models/embeddings.joblib "https://huggingface.co/Elthon5/modelos/resolve/main/embeddings.joblib"
wget --header="Authorization: Bearer $HF_TOKEN" -O Models/nn_model.joblib "https://huggingface.co/Elthon5/modelos/resolve/main/nn_model.joblib"