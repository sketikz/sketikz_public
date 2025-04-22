import streamlit as st
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# Inference steps taken from https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

@st.cache_resource
def get_model(model_path):
    try:
        with st.spinner(f"Loading model {model_path}"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Load the model here
            model_import = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map=device
            )
            processor_import = AutoProcessor.from_pretrained(model_path)

            return model_import, processor_import
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def run_inference(input_file, model_path, args):
    model, processor = get_model(model_path)
    if model is None or processor is None:
        return "Error loading model."
    image = Image.open(input_file)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Please generate TikZ code to draw the diagram of the given image."}
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image, text_prompt, return_tensors="pt").to("cuda")

    output_ids = model.generate(**inputs,
                                max_new_tokens=args.max_length,
                                do_sample=True,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                num_return_sequences=1,
                                temperature=args.temperature
                            )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text
