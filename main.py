import streamlit as st
from transformers import pipeline

@st.cache_resource
def get_model():
    # Load the model here
    model = pipeline("image-to-text", model="itsumi-st/imgtikz_qwen2vl")
    return model

st.logo("NLP_Group_logo.svg", size="large")
main_page = st.Page("main_page.py", title="Main Page", icon="🏠")
sketch2diagram_page = st.Page("sketch2diagram.py", title="Sketch2Diagram", icon="🖼️")
# Add pages to the main page

pg = st.navigation([main_page, sketch2diagram_page])

pg.run()