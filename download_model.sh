#!/bin/bash
mkdir -p Models
wget -O Models/llama-2-7b-chat.Q2_K.gguf "https://huggingface.co/Elthon5/dipro-agent-bot-embeddings/resolve/main/llama-2-7b-chat.Q2_K.gguf"
wget -O Models/embeddings.joblib "https://huggingface.co/Elthon5/dipro-agent-bot-embeddings/resolve/main/embeddings.joblib"
wget -O Models/nn_model.joblib "https://huggingface.co/Elthon5/dipro-agent-bot-embeddings/resolve/main/nn_model.joblib"