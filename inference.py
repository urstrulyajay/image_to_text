import torch
from unsloth import FastVisionModel
from PIL import Image
import requests
from io import BytesIO
import json
import numpy as np
import gradio as gr

# Load config
with open("config.json") as f:
    config = json.load(f)

# Load model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(config["save_dir"])
model.eval()

# Load and resize image
def load_image(url, image_size):
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        image = Image.new("RGB", (image_size, image_size))
    image = image.resize((image_size, image_size))
    image_np = np.array(image).astype(np.uint8)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    return image_tensor

# Prediction function
def predict(image_url):
    instruction = config["instruction"]
    image_tensor = load_image(image_url, config["image_size"])
    sample = {
        "image": image_tensor,
        "text": ""
    }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]}
            ]
        }
    ]

    output = model.chat(tokenizer, messages=messages)
    return output

# Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Image URL"),
    outputs=gr.Textbox(label="Extracted Nutrition JSON"),
    title="Nutrition Extractor",
    description="Enter the URL of an image containing nutrition information to extract its contents using the trained model."
)

if __name__ == "__main__":
    iface.launch()
