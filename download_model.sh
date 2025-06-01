#!/bin/bash
set -e
mkdir -p Models
wget -O Models/llama-2-7b-chat.Q2_K.gguf "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf"