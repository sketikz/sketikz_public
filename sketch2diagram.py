import streamlit as st
from PIL import Image

from main import get_model

# Sidebar Setup
st.sidebar.title("Model Configuration")
inference_strat = st.sidebar.selectbox("Inference Strategy", ["Iterative", "Multi-candidate"],
                                       help="Choose the inference strategy for the model. Iterative generates one candidate at a time until an output compiles, while Multi-candidate generates multiple candidates in parallel.")

# Introduction Section
st.title("Sketch2Diagram")

st.write("This is a runnable demo of ImgTikZ model introduced in the Sketch2Diagram paper.")
st.write("Please refer to the [original paper](https://openreview.net/pdf?id=KvaDHPhhir) for more details.")
st.write("The model is trained to convert sketches into TikZ code, which can be used to generate vectorized diagrams.")
st.write(f"Inference Strategy: {inference_strat}")

# User Input Section
st.subheader("Upload your sketch")

input_method = st.selectbox("Input Method", ["Upload", "Camera"],
                            help="Choose how you want to input your sketch. You can either upload an image or take a picture using your webcam.")

input_file = None
if input_method == "Camera":
    input_file = st.camera_input("Take a picture of your sketch")
    # Implement camera input functionality here
else:
    input_file = st.file_uploader("Upload an image of your sketch", type=["png", "jpg", "jpeg"])

generate_command = None
# Display the uploaded image
if input_file is not None:
    st.image(input_file, caption="Uploaded Sketch")
    generate_command = st.button("Generate TikZ Code")

if generate_command:
    model = get_model()
    image = Image.open(input_file)
    with st.spinner("Generating TikZ code..."):
        output = model(image)

    tikz_code = output[0]['generated_text']
    st.subheader("Generated TikZ Code")
    st.code(tikz_code, language='latex')
