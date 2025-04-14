import streamlit as st
import numpy as np
from PIL import Image

### App de Inicio

def createPage():
    
    # Title of the main page
    pathLogo = pathLogo = r'D:\Proyectos\DIPRO Agent BOT\Imagenes\Dipro_Logo1.png'
    # Abrir imagen y convertirla a RGB (3 canales)
     # Abrir imagen con canal alfa
    img = Image.open(pathLogo).convert("RGBA")
    # Convertir a numpy array
    #display = np.array(display)
    col1, col2, col3 = st.columns([1.5,1,1])
    with col2:
        st.image(img, use_container_width=False)

    st.markdown('Bienvenido a ***Aplicación DIPRO***, para identificar patrones semánticos en las preguntas del FDI y sus respuestas para encontrar relaciones semánticas entre los registros.')

    st.write(""" 
    Está aplicación contiene:
    + ***Análisis, interpretación y agrupación de las “Pregunta del RFI”, por tipo de consulta técnica elaborada.***
    + ***Toma de decisión si las consultas son por omisión del proyecto, por error en el diseño del proyecto, por no haber reflejado la información correcta en el plano correcto u otros.***
    + ***Identificar el 80/20 de las consultas que más suceden, así como la causa raíz.***
    + ***Chatbot para Atender Preguntas del FDI.***
    """)

    return True