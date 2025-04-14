import streamlit as st
import os
from PIL import Image
from streamlit_option_menu import option_menu
import start, analisis, chatbot # Importar páginas acá

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas relativas portables

FAVICON_PATH = os.path.join(BASE_DIR, "Imagenes", "Dipro_Logo.ico")

 #### Páginas
im = Image.open(FAVICON_PATH)
st.set_page_config(page_title='DIPRO', page_icon=im, layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

v_menu=["Inicio", "Análisis", "Chatbot"]

selected = option_menu(
    menu_title=None,  # required
    options=["Inicio", "Análisis", "Chatbot"],  # required 
    icons=["house", "graph-up", "robot"],  # optional
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
    styles={
        "container": {"padding": "10px", "background-color": "#fafafa"},
        "icon": {"font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "center", "margin":"0px", "--hover-color": "salmon"},
        "nav-link-selected": {"background-color": "tomato"},
    }
    )

if selected=="Inicio":
    start.createPage()

if selected=="Análisis":
    analisis.createPage()

if selected=="Chatbot":
    chatbot.createPage()