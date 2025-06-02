#!/bin/bash
set -e
mkdir -p Models

echo "Descargando llama-2-7b-chat.Q2_K.gguf..."
wget --header="Authorization: Bearer $HF_TOKEN" -O Models/llama-2-7b-chat.Q2_K.gguf "https://huggingface.co/Elthon5/dipro-agent-bot-embeddings/resolve/main/llama-2-7b-chat.Q2_K.gguf" || { echo "Error descargando llama-2-7b-chat.Q2_K.gguf"; exit 1; }

echo "Descargando embeddings.joblib..."
wget --header="Authorization: Bearer $HF_TOKEN" -O Models/embeddings.joblib "https://huggingface.co/Elthon5/dipro-agent-bot-embeddings/resolve/main/embeddings.joblib" || { echo "Error descargando embeddings.joblib"; exit 1; }

echo "Descargando nn_model.joblib..."
wget --header="Authorization: Bearer $HF_TOKEN" -O Models/nn_model.joblib "https://huggingface.co/Elthon5/dipro-agent-bot-embeddings/resolve/main/nn_model.joblib" || { echo "Error descargando nn_model.joblib"; exit 1; }

echo "Descarga completada."