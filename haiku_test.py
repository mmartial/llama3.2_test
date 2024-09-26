# Adapted from: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

import os
import sys
from dotenv import load_dotenv

def error_exit(text):
    print(text)
    sys.exit(1)

def check_file_r(file):
    if not os.path.exists(file):
        error_exit(f"{file} does not exist")
    if not os.access(file, os.R_OK):
        error_exit(f"file ({file}) can not be read")
    return ""

# Place your HF_TOKEN in a .env file in the same directory as this script
# How to get your token: https://huggingface.co/docs/hub/en/security-tokens
err = check_file_r(".env")
if err == "":
    load_dotenv()

hf_token = ''
if 'HF_TOKEN' in os.environ:
    hf_token = os.environ.get('HF_TOKEN')

model_id = "meta-llama/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=hf_token,
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
inputs = processor(image, prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
