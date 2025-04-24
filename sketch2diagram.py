import streamlit as st
from pdf2image import convert_from_path

from qwen2_inference import run_inference
from util import compile_tikz_to_pdf

args = {}

# Sidebar Setup
st.sidebar.title("Model Configuration")
model_name = st.sidebar.selectbox("Model Name", ['Itsumi-st/Imgtikz_Qwen2vl', 'Qwen/Qwen2-VL-7B-Instruct'])
args['inference_strat'] = st.sidebar.selectbox("Inference Strategy", ["Iterative", "Multi-candidate"],
                                               help="Choose the inference strategy for the model. Iterative generates one candidate at a time until an output compiles, while Multi-candidate generates multiple candidates in parallel.")
args['max_length'] = st.sidebar.slider("Max Length", 1, 5096, 2048,
                                       help="Maximum length of the generated output. The model will generate text up to this length.")
args['seed'] = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)
args['temperature'] = st.sidebar.slider("Temperature", 0.0, 1.0, 0.6, step=0.01,
                                        help="Temperature parameter for sampling. Higher values result in more random outputs.")
args['top_p'] = st.sidebar.slider("Top P", 0.0, 1.0, 1.0, step=0.01,
                                  help="Top P sampling parameter. The model will sample from the top P percentage of the probability distribution.")
args['top_k'] = st.sidebar.slider("Top K", 0, 100, 50, step=1,
                                  help="Top K sampling parameter. The model will sample from the top K tokens with the highest probabilities.")

# Introduction Section
st.title("Sketch2Diagram")

st.write("This is a runnable demo of ImgTikZ model introduced in the Sketch2Diagram paper.")
st.write("Please refer to the [original paper](https://openreview.net/pdf?id=KvaDHPhhir) for more details.")
st.write("The model is trained to convert sketches into TikZ code, which can be used to generate vectorized diagrams.")

# User Input Section
st.subheader("Upload your sketch")

input_method = st.selectbox("Input Method", ["Upload", "Camera"],
                            help="Choose how you want to input your sketch. You can either upload an image or take a picture using your webcam.")

input_file = None
if input_method == "Camera":
    input_file = st.camera_input("Take a picture of your sketch")
    # todo: Implement camera input functionality here
else:
    input_file = st.file_uploader("Upload an image of your sketch", type=["png", "jpg", "jpeg"])
st.write(args)
generate_command = None
# Display the uploaded image
if input_file is not None:
    st.image(input_file, caption="Uploaded Sketch")
    generate_command = st.button("Generate TikZ Code")

# Run model inference
if generate_command:
    with st.spinner("Generating TikZ code..."):
        output = run_inference(input_file, model_name, args)[0]
        pdf_file_path = compile_tikz_to_pdf(output)
        if output and pdf_file_path:
            st.success("TikZ code generated successfully!")
            st.code(output, language='latex')

            st.download_button(
                label="Download LaTeX Code",
                data=output,
                file_name="output.tex",
                mime="text/plain"
            )

            # st.image(pdf_file_path, caption="Generated Diagram", use_column_width=True)
            with open(pdf_file_path, "rb") as f:
                st.download_button(
                    label="Download PDF",
                    data=f.read(),  # ✅ this is the binary content
                    file_name="output.pdf",
                    mime="application/pdf"
                )

            images = convert_from_path(pdf_file_path)
            st.image(images[0], caption="Generated Diagram", use_column_width=True)
        else:
            st.error("Failed to generate TikZ code.")
