import os

import streamlit as st
import torch
from PIL import Image
from dotenv import load_dotenv
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")


def print_gpu_memory(label, memory_allocated, memory_reserved):
    if torch.cuda.is_available():
        print("-----------------------------------")
        print(f"{label} GPU Memory Usage:")
        print(f"Allocated: {memory_allocated / 1024 ** 2:.2f} MB")
        print(f"Cached: {memory_reserved / 1024 ** 2:.2f} MB")


# Inference steps taken from https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

# @st.cache_resource
def get_model(model_path):
    try:
        with st.spinner(f"Loading model {model_path}"):
            # Load the model here
            model_import = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto",
                attn_implementation="flash_attention_2",
                token=HUGGINGFACE_TOKEN,
            )
            model_import = model_import.to("cuda")
            size = {
                "shortest_edge": 224,
                "longest_edge": 1024,
            }
            processor_import = AutoProcessor.from_pretrained("itsumi-st/imgtikz_qwen2vl",
                                                             size=size,
                                                             min_pixels=256 * 256,
                                                             max_pixels=1024 * 1024,
                                                             token=HUGGINGFACE_TOKEN)
            processor_import.tokenizer.padding_side = 'left'

            return model_import, processor_import
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def run_inference(input_file, model_path, args):
    model, processor = get_model(model_path)
    if model is None or processor is None:
        return "Error loading model."

    # GPU Memory after model loading:
    after_model_dump = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

    image = Image.open(input_file)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Please generate TikZ code to draw the diagram of the given image."}
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_input, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[text_prompt],
        images=image_input,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # GPU Memory after input processing
    after_input_dump = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

    output_ids = model.generate(**inputs,
                                max_new_tokens=args['max_length'],
                                do_sample=True,
                                top_p=args['top_p'],
                                top_k=args['top_k'],
                                use_cache=True,
                                num_return_sequences=1,
                                pad_token_id=processor.tokenizer.pad_token_id,
                                temperature=args['temperature']
                                )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # GPU Memory after generation
    after_gen_dump = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

    print_gpu_memory("Before Model", after_model_dump[0], after_model_dump[1])
    print_gpu_memory("After Input", after_input_dump[0], after_input_dump[1])
    print_gpu_memory("After Generation", after_gen_dump[0], after_gen_dump[1])

    return output_text
