#!/bin/bash
set -e
mkdir -p Models
wget --header="Authorization: Bearer $HF_TOKEN" -O Models/llama-2-7b-chat.Q2_K.gguf "https://huggingface.co/Elthon5/dipro-agent-bot-embeddings/resolve/main/llama-2-7b-chat.Q2_K.gguf"
wget --header="Authorization: Bearer $HF_TOKEN" -O Models/embeddings.joblib "https://huggingface.co/Elthon5/dipro-agent-bot-embeddings/resolve/main/embeddings.joblib"
wget --header="Authorization: Bearer $HF_TOKEN" -O Models/nn_model.joblib "https://huggingface.co/Elthon5/dipro-agent-bot-embeddings/resolve/main/nn_model.joblib"