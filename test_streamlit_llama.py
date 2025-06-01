import streamlit as st
from llama_cpp import Llama

@st.cache_resource
def get_llm():
    return Llama(model_path="Models/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=256)

st.title("Prueba Llama en Streamlit")
if st.button("Probar modelo"):
    llm = get_llm()
    respuesta = llm("¿Cuál es la capital de Francia?", max_tokens=10)
    st.write(respuesta['choices'][0]['text'])