import os

import streamlit as st
from PIL import Image

logo_path = os.path.join(os.path.dirname(__file__), "NLP_Group_logo.png")
logo = Image.open(logo_path)
st.logo(logo, size="large")
main_page = st.Page("main_page.py", title="Main Page", icon="🏠")
sketch2diagram_page = st.Page("sketch2diagram.py", title="Sketch2Diagram", icon="🖼️")
# Add pages to the main page

pg = st.navigation([main_page, sketch2diagram_page])

pg.run()