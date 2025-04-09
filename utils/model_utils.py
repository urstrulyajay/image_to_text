import json
import requests
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms

class ImageTextJSONDataset(Dataset):
    def __init__(self, json_path, image_size):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_size = image_size
        self.resizer = transforms.Resize((image_size, image_size))

    def __len__(self):
        return len(self.data)

    def download_image(self, url):
        try:
            response = requests.get(url, timeout=5)
            return Image.open(BytesIO(response.content)).convert('RGB')
        except:
            return Image.new("RGB", (self.image_size, self.image_size))

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.download_image(item["image_nutrition_url"])
        image = self.resizer(image)
        image_np = np.array(image).astype(np.uint8)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return {
            "image": image_tensor,
            "text": item["nutrition_json"]
        }

def convert_to_conversation(sample, instruction):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["text"]}
            ]
        },
    ]
    return {"messages": conversation}
